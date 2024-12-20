import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
import torch.utils.data.sampler as torch_sampler
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from loss import *
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import eval_metrics as em
from ecapa_tdnn import *
from speechformer import SpeechFormer
from lcnn_speechformer import LCNNSF
torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

def initParams(): 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", '--name', type=str, help="datasets name", default='')
    parser.add_argument('--upsample_num', type=int, help="real utterance augmentation number", default=0)
    parser.add_argument('--downsample_num', type=int, help="fake utterance augmentation number", default=0)
    
    parser.add_argument('--seed', type=int, help="random number seed", default=688)

    # Data folder prepare
    parser.add_argument("-a", "--access_type", type=str, help="LA or PA", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/home/chenghaonan/xieyuankun/data/asv2019')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/home/chenghaonan/xieyuankun/data/asv2019/preprocess_xls-r')

    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/model/')

    parser.add_argument("--ratio", type=float, default=1,
                        help="ASVspoof ratio in a training batch, the other should be augmented")

    # Dataset prepare
    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r')
    parser.add_argument("--feat_len", type=int, help="features length", default=201)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=False,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='repeat', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=256)

    parser.add_argument('-m', '--model', help='Model arch', default='lcnn',
                        choices=['cnn', 'resnet', 'lcnn', 'res2net', 'ecapa','speechformer','lcnnsf'])
    parser.add_argument('--add_loss', type=str, default=None)
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="3")
    parser.add_argument("--multigpu", type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2, help="number of workers")

    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")

    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")
    parser.add_argument('--lambda_', type=float, default=0.05, help="lambda for gradient reversal layer")
    parser.add_argument('--pre_train', action='store_true', help="whether to pretrain the model")
    parser.add_argument('--test_on_eval', action='store_true',
                        help="whether to run EER on the evaluation set")

    args = parser.parse_args()



    # Change this to specify GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")
        with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
            file.write("Start recording test loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'resnet':
        node_dict = {"LFCC": 3}
        feat_model = ResNet(3, args.enc_dim, resnet_type='18',
                            nclasses=1 if args.base_loss == "bce" else 2).to(args.device)
    elif args.model == 'lcnn':
        feat_model = LCNN(201, args.enc_dim, nclasses=2).to(args.device)
    elif args.model == 'ecapa':
        node_dict = {"LFCC": 60}
        feat_model = Res2Net2(Bottle2neck, C=512, model_scale=8, nOut=2, n_mels=node_dict[args.feat]).to(args.device)
    elif args.model == 'res2net':
        feat_model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=False, num_classes=2).to(
            args.device)
    elif args.model =='speechformer':
        feat_model = SpeechFormer(input_dim=201, ffn_embed_dim=2048, num_classes=2, hop=0.01, num_layers=[2, 2, 4, 4],
                           expand=[1, 1, 1, -1], num_heads=128, dropout=0.1, attention_dropout=0.1).to(args.device)

    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    if args.multigpu:
        feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)

    training_set = RawAudio(path_to_features=f'./preprocess_xls-r_{args.name}/xls-r',
    path_to_meta=f'../datasets/{args.name}', meta_file='meta.csv', pad_chop=False, part='train')

    # 分離真實人聲和假人聲
    meta = pd.read_csv(f'../datasets/{args.name}/train/meta.csv')
    real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
    spoof_indices = meta[meta['label'] == 'spoof'].index.tolist()
    
    # 如果需要下採樣假人聲
    if args.downsample_num > 0 and len(spoof_indices) > args.downsample_num:
        selected_spoof_indices = random.sample(spoof_indices, args.downsample_num)
    else:
        selected_spoof_indices = spoof_indices  # 如果不需要下採樣，保留所有假人聲

    # 新的數據集(下採樣假人聲)
    real_subset = Subset(training_set, real_indices)
    spoof_subset = Subset(training_set, selected_spoof_indices)
    training_set = ConcatDataset([real_subset, spoof_subset])
    
    # 如果需要從 LibriSpeech 上採樣真實人聲
    if args.upsample_num > 0:
        training_set_real_utterance =  RawAudio(path_to_features=f'./preprocess_xls-r_LibriSpeech/xls-r'
                             , path_to_meta = f'../datasets/LibriSpeech'
                             , meta_file='meta.csv', pad_chop=False, part='train')
        selected_bonafide_indices = random.sample(range(len(training_set_real_utterance)), args.upsample_num)
        bonafide_subset =  Subset(training_set_real_utterance, selected_bonafide_indices)
        training_set = ConcatDataset([training_set, bonafide_subset])
        
    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                                    shuffle=True, num_workers=args.num_workers)
    trainOri_flow = iter(trainOriDataLoader)

    validation_set = RawAudio(path_to_features=f'./preprocess_xls-r_{args.name}/xls-r'
                             , path_to_meta = f'../datasets/{args.name}'
                             , meta_file='meta.csv', pad_chop=False, part='validation')
    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size * args.ratio),
                                  shuffle=True, num_workers=args.num_workers)
    valOri_flow = iter(valOriDataLoader)

    weight = torch.FloatTensor([10, 1]).to(args.device)
    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.functional.binary_cross_entropy()

    early_stop_cnt = 0
    add_size = args.batch_size - int(args.batch_size * args.ratio)
    prev_loss = 1e8

    monitor_loss = 'base_loss'


    for epoch_num in tqdm(range(args.num_epochs)):
        genuine_feats, ip1_loader, idx_loader = [], [], []
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        print('\nEpoch: %d ' % (epoch_num + 1))
        correct_m, total_m, correct_c, total_c, correct_v, total_v = 0, 0, 0, 0, 0, 0

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                featOri, labelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                featOri, labelsOri = next(trainOri_flow)

            feat = featOri
            labels = labelsOri
            feat = feat.transpose(2, 3).to(args.device)
            labels = labels.to(args.device)
            feat, labels = shuffle(feat, labels)

            if args.model == 'ecapa':
                feat = torch.squeeze(feat)
            print(feat.shape)
            #feat = feat.squeeze(dim=1)
            feats, feat_outputs = feat_model(feat)
            print(feat_outputs.shape)
            if args.base_loss == "bce":
                feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
            else:
                feat_loss = criterion(feat_outputs, labels)
            trainlossDict['base_loss'].append(feat_loss.item())
            if args.add_loss == None:
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(trainlossDict[monitor_loss][-1]) + "\n")

        feat_model.eval()
        with torch.no_grad():
            ip1_loader, idx_loader, score_loader = [], [], []
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    featOri, labelsOri= next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    featOri, labelsOri= next(valOri_flow)

                feat = featOri
                labels = labelsOri
                feat = feat.transpose(2, 3).to(args.device)
                labels = labels.to(args.device)
                feat, labels = shuffle(feat, labels)
                if args.model == 'ecapa':
                    feat = torch.squeeze(feat)
                print(feat.shape)
                feats, feat_outputs = feat_model(feat)
                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                ip1_loader.append(feats)
                idx_loader.append((labels))

                if args.add_loss in [None]:
                    devlossDict["base_loss"].append(feat_loss.item())

                score_loader.append(score)

                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                print(desc_str)
                scores = torch.cat(score_loader, 0).data.cpu().numpy()
                labels = torch.cat(idx_loader, 0).data.cpu().numpy()
                eer, frr, far, threshold = em.compute_eer(scores[labels == 0], scores[labels == 1])
                other_eer, frr, far, other_eer_threshold = em.compute_eer(-scores[labels == 0], -scores[labels == 1])
                eer = min(eer, other_eer)

                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" +
                              str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                              str(eer) +"\n")
                print("Val EER: {}".format(eer))

        valLoss = np.nanmean(devlossDict[monitor_loss])
        if (epoch_num + 1) % 1 == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))
            loss_model = None

        if valLoss < prev_loss:
            # Save the model checkpoint
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            loss_model = None
            prev_loss = valLoss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 500:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 499))
            break
        # if early_stop_cnt == 1:
        #     torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')
    return feat_model, loss_model


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)

