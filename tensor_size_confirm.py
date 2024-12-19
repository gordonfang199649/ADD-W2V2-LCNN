import torch

# 假設有一個示例 .pt 文件路徑
filepath = "./preprocess_xls-r_in_the_wild/xls-r/00000_0_Alec Guinness_spoof.pt"

# 加載張量
featureTensor = torch.load(filepath, weights_only=False)
print("張量大小:", featureTensor.shape)

filepath = "./preprocess_xls-r/eval/xls-r/00000_LA_E_2834763_A11_spoof.pt"

# 加載張量
featureTensor = torch.load(filepath, weights_only=False)
print("張量大小:", featureTensor.shape)
