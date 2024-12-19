import raw_dataset as dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)
device = torch.device("cuda:1" if cuda else "cpu")
def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform
    
def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data

def extract_representation(path_to_database, dataset_name):
    for part_ in ["test"]:
        raw_audio = dataset.AudioRawDataSet(path_to_database=path_to_database, meta_csv='meta.csv', part=part_)
        target_dir = os.path.join(f"./preprocess_xls-r_{dataset_name}","xls-r", part_)
        processor =  Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
        #model.eval()
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for idx in tqdm(range(len(raw_audio))):
            waveform, filen_ame, label = raw_audio[idx]
            waveform = waveform.to(device)
            waveform = pad_dataset(waveform).to('cpu')
            input_values = processor(waveform, sampling_rate=16000,return_tensors="pt").input_values.cuda()
            with torch.no_grad():
                wav2vec2 = model(input_values).last_hidden_state.cuda()
            print(wav2vec2.shape)
            torch.save(wav2vec2, os.path.join(target_dir, "%s.pt" % (filen_ame)))
        print(f"Done with {part_} sets from {dataset_name}")

if __name__ == '__main__':
    extract_representation(path_to_database='../datasets/DFADD', dataset_name='DFADD')
    extract_representation(path_to_database='../datasets/ASVspoof2021_DF', dataset_name='ASVspoof2021_DF')
    extract_representation(path_to_database='../datasets/CodecFake', dataset_name='CodecFake')
    extract_representation(path_to_database='../datasets/in_the_wild', dataset_name='in_the_wild')
    extract_representation(path_to_database='../datasets/accentdb', dataset_name='accentdb')