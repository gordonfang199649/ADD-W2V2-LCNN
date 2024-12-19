#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
from typing import Tuple
import soundfile as sf
import pandas as pd

torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    wave, sr = sf.read(filepath)
    
    if len(wave.shape) == 2 and wave.shape[1] == 2:
        wave = wave.mean(axis=1)  # 雙聲道取均值
        
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

class ASVspoof2019Raw(Dataset):
    def __init__(self, access_type, path_to_database, path_to_protocol, part='train'):
        super(ASVspoof2019Raw, self).__init__()
        self.access_type = access_type
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.path_to_protocol = path_to_protocol
        if self.part =='train':
            protocol = os.path.join(os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trn.txt'))
        else:
            protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        # if self.part == "eval":
        #     protocol = os.path.join(self.ptd, access_type, 'ASVspoof2019_' + access_type +
        #                             '_cm_protocols/ASVspoof2019.' + access_type + '.cm.' + self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        # # would not work if change data split but this csv is only for feat_len
        # self.csv = pd.read_csv(self.ptf + "Set_csv.csv")

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + ".flac")
        waveform, sr = torchaudio_load(filepath)

        return waveform, filename, tag, label

    def collate_fn(self, samples):
        return default_collate(samples)

class AudioRawDataSet(Dataset):
    def __init__(self, path_to_database, meta_csv, part='train') -> None:
        super().__init__()
        self.path_to_audio = path_to_database
        self.label = {"spoof": 1, "bonafide": 0}
        self.part = part
         # 讀取 meta.csv 文件
        meta_path = os.path.join(path_to_database, part, meta_csv)
        self.meta_data = pd.read_csv(meta_path)

    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        # 從 meta.csv 中獲取文件名、說話者和標籤
        file_name = self.meta_data.iloc[idx]['filename']
        label = self.meta_data.iloc[idx]['label']
        
        # 構建文件的完整路徑
        filepath = os.path.join(self.path_to_audio, self.part, 'audio', file_name)
        waveform, _ = torchaudio_load(filepath)
        
        return waveform, file_name, label

        
    def collate_fn(self, samples):
        return default_collate(samples)