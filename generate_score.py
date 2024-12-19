import random
from dataset import *
from model import *
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch
import os
from tqdm import tqdm
import argparse
import json
import eval_metrics as em
import matplotlib.pyplot as plt

torch.multiprocessing.set_start_method('spawn', force=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument('-n', '--model_name', type=str, help="the name of the model",
                        required=False, default='w2v2_LCNN')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score",
                        default='./scores')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would liek to score on",
                        required=False, default='19eval')
    parser.add_argument("--gpu", type=str, help="GPU index", default="2")
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if '19' in args.task:
        args.out_score_dir = "./scores"
    else:
        args.out_score_dir = args.score_dir

    return args

def test_on_desginated_datasets(task, feat_model_path, output_score_path, model_name):
    # 加載模型
    model = torch.load(feat_model_path).to(device)
    model.eval()

    # 加載數據集
    test_set = RawAudio(path_to_features=f'./preprocess_xls-r_{task}/xls-r',
    path_to_meta=f'../datasets/{task}', meta_file='meta.csv', pad_chop=False, part='test')
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    # 用於保存結果
    score_loader = []
    label_loader = []

    # 遍歷測試數據
    for i, data_slice in enumerate(tqdm(testDataLoader)):
        waveforms, labels = data_slice
        waveforms = waveforms.transpose(2, 3).to(device)
        labels = labels.to(device)

        # 模型推理
        _, model_outputs = model(waveforms)
        score = F.softmax(model_outputs, dim=1)[:, 0]

        # 保存分數和標籤
        score_loader.append(score.item())
        label_loader.append(labels.item())

    # 計算 EER
    # 轉換為 numpy 數組
    scores = np.array(score_loader)
    labels = np.array(label_loader)

    # 根據標籤分割分數
    target_scores = scores[labels == 0]  # 正例 (bonafide)
    nontarget_scores = scores[labels == 1]  # 負例 (spoof)

    eer, frr, far, threshold = em.compute_eer(target_scores, nontarget_scores)
    print(f'Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}, False Acceptance Rate (FAR): {far}, Threshold: {threshold}')

if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    test_on_desginated_datasets(args.task, model_path, args.score_dir, args.model_name)