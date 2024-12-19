import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataset import RawAudio
from torch.utils.data import DataLoader, Subset
from sklearn.utils import resample
from tqdm import tqdm
from sklearn.decomposition import PCA

def balance_indices(labels, target_size, real_to_fake_ratio):
    """
    根據目標大小和比例，計算平衡後的索引。
    """
    real_indices = np.where(labels == 0)[0]
    fake_indices = np.where(labels == 1)[0]

    # 計算目標數量
    real_target = int(target_size * real_to_fake_ratio / (1 + real_to_fake_ratio))
    fake_target = target_size - real_target

    # 欠採樣或過採樣
    real_indices_balanced = resample(real_indices, replace=len(real_indices) < real_target, n_samples=real_target, random_state=42)
    fake_indices_balanced = resample(fake_indices, replace=len(fake_indices) < fake_target, n_samples=fake_target, random_state=42)

    # 合併索引
    balanced_indices = np.concatenate([real_indices_balanced, fake_indices_balanced])
    np.random.shuffle(balanced_indices)
    return balanced_indices

def main():
    model_names = ["DFADD", "CodecFake", "ASVspoof2021_DF", "in_the_wild"]
    all_features, all_labels, all_model_names = [], [], []
    target_size = 20000  # 每個資料集大小
    real_to_fake_ratio = 1 / 2  # 真實與假聲音比例

    for model_name in model_names:
        # 初始化數據集
        dataset = RawAudio(
            path_to_features=f'./preprocess_xls-r_{model_name}/xls-r',
            path_to_meta=f'../datasets/{model_name}',
            meta_file='meta.csv',
            pad_chop=False,
            part='test'
        )

        # 計算平衡後的索引
        all_labels_np = np.array([dataset[i][1] for i in range(len(dataset))])  # 提取所有標籤
        balanced_indices = balance_indices(all_labels_np, target_size, real_to_fake_ratio)

        # 創建子集
        balanced_dataset = Subset(dataset, balanced_indices)
        data_loader = DataLoader(balanced_dataset, batch_size=1, shuffle=True, num_workers=0)

        # 提取特徵與標籤
        for features, labels in tqdm(data_loader, desc=f"Processing {model_name}"):
            features_flatten = features.view(features.size(0), -1).cpu().numpy()  # 保持每個樣本為 2D
            all_features.append(features_flatten)
            all_labels.append(labels.cpu().numpy())
            all_model_names.extend([model_name] * labels.size(0))

    # 合併資料
    all_features = np.concatenate(all_features, axis=0)  # 正確拼接成 [樣本數, 特徵數]
    all_labels = np.concatenate(all_labels, axis=0)

    pca = PCA(n_components=50)
    all_features = pca.fit_transform(all_features)

    # t-SNE 降維
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, learning_rate=200, n_iter=1000)
    tsne_results = tsne.fit_transform(all_features)

    # 繪製
    plt.figure(figsize=(10, 8))
    color_map = {"DFADD": "red", "CodecFake": "blue", "ASVspoof2021_DF": "green", "in_the_wild": "purple", "real": "black"}

    for model_name in model_names:
        fake_indices = [i for i, (name, label) in enumerate(zip(all_model_names, all_labels)) if name == model_name and label == 1]
        plt.scatter(tsne_results[fake_indices, 0], tsne_results[fake_indices, 1], label=f"{model_name} (Fake)", alpha=0.7, color=color_map[model_name])

    real_indices = [i for i, label in enumerate(all_labels) if label == 0]
    plt.scatter(tsne_results[real_indices, 0], tsne_results[real_indices, 1], label="Real", alpha=0.7, color=color_map["real"])

    plt.title("t-SNE Visualization of Real vs Synthetic Audio Representations")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.savefig('./wav2vec_distribution_in_t-SNE.png')

if __name__ == "__main__":
    main()
