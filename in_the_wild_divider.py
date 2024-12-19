import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Path
dataset_dir = "./release_in_the_wild"
meta_file = os.path.join(dataset_dir, "meta.csv")
output_dirs = {
    "train": os.path.join(dataset_dir, "train"),
    "eval": os.path.join(dataset_dir, "eval"),
    "test": os.path.join(dataset_dir, "test"),
}

# create new folders
for subset in output_dirs.values():
    os.makedirs(subset, exist_ok=True)

# read meta data
meta_data = pd.read_csv(meta_file)

# label distribution
print("Original dataset distribution:")
print(meta_data["label"].value_counts())

# divide real and fake audio
bona_fide = meta_data[meta_data["label"] == "bona-fide"]
spoof = meta_data[meta_data["label"] == "spoof"]

# define ratio
train_ratio, eval_ratio, test_ratio = 0.7, 0.1, 0.2

# split real and fake audio by ratio
def split_data(data, train_ratio, eval_ratio, test_ratio):
    train, temp = train_test_split(data, test_size=(1 - train_ratio), random_state=42, shuffle=True)
    eval_set, test_set = train_test_split(temp, test_size=(test_ratio / (eval_ratio + test_ratio)), random_state=42, shuffle=True)
    return train, eval_set, test_set

bona_train, bona_eval, bona_test = split_data(bona_fide, train_ratio, eval_ratio, test_ratio)
spoof_train, spoof_eval, spoof_test = split_data(spoof, train_ratio, eval_ratio, test_ratio)

# merge real and fake audio for subsets
train_set = pd.concat([bona_train, spoof_train]).sample(frac=1, random_state=42)  # 打亂
eval_set = pd.concat([bona_eval, spoof_eval]).sample(frac=1, random_state=42)  # 打亂
test_set = pd.concat([bona_test, spoof_test]).sample(frac=1, random_state=42)  # 打亂

# Subset distribution
print("Training set distribution:")
print(train_set["label"].value_counts())
print("Evaluation set distribution:")
print(eval_set["label"].value_counts())
print("Test set distribution:")
print(test_set["label"].value_counts())

# save meta.csv
def save_subset(subset_data, subset_dir):
    meta_path = os.path.join(subset_dir, "meta.csv")
    subset_data.to_csv(meta_path, index=False)  # 保存 meta.csv
    for _, row in subset_data.iterrows():
        src = os.path.join(dataset_dir, row["file"])
        dest = os.path.join(subset_dir, row["file"])
        if os.path.exists(src):
            shutil.copy(src, dest)

# save subsets
save_subset(train_set, output_dirs["train"])
save_subset(eval_set, output_dirs["eval"])
save_subset(test_set, output_dirs["test"])

print("Data splitting completed. Subsets saved in:")
for subset, path in output_dirs.items():
    print(f"{subset}: {path}")
