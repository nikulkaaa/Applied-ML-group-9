import pickle
from pathlib import Path
from PIL import Image
import numpy as np

from src.image_handler import DatasetHandler
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt


def make_pca(pca_path: Path) -> None:
    pca = IncrementalPCA(n_components=200)
    print('made PCA')

    for batch in new_DH():
        #print("Any NaNs:", np.isnan(batch).any())
        #print("Any Infs:", np.isinf(batch).any())
        #variances = np.var(batch, axis=0)
        #print("Zero variance features:", np.sum(variances == 0))

        pca.partial_fit(batch)
    print('fit PCA')

    with open(pca_path, 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)

def new_DH() -> DatasetHandler:
    DH = DatasetHandler()
    DH.load_images(grayscale=True) # only loads images if they have not been written to a binary file yet
    return DH

def print_count_per_label(DH: DatasetHandler) -> None:

    print("Real count:", np.sum(DH.labels == 1))
    print("Fake count:", np.sum(DH.labels == 0))

def plot_PCA(pca_path: Path) -> None:
    with open(pca_path, 'rb') as file:
        pca = pickle.load(file)
        # plot PCA variance with 200 components
        pca_images = []
        DH = new_DH()
        for batch in DH:
            pca_images.extend(pca.transform(batch))

        print(len(pca_images))
        pca_images = np.array(pca_images)
        plt.figure(figsize=(8, 6))

        print(len(pca_images), len(DH.labels))

        fake_variance = np.var(pca_images[DH.labels == 0], axis=0).mean()
        real_variance = np.var(pca_images[DH.labels == 1], axis=0).mean()

        print("Real count:", np.sum(DH.labels == 1))
        print("Fake count:", np.sum(DH.labels == 0))

        plt.scatter(pca_images[DH.labels == 1, 0], pca_images[DH.labels == 1, 1],
                    color='red', label=f'Real (mean variance: {real_variance:.2f}', alpha=0.9)
        # plt.xlim(-3020, -3000)
        # plt.ylim(-68, -67.5)
        plt.scatter(pca_images[DH.labels == 0, 0], pca_images[DH.labels == 0, 1],
                    color='blue', label=f'Fake (mean variance: {fake_variance:.2f}', alpha=0.9)

        plt.xlabel(f'PCA Component 1 (explained variance: {pca.explained_variance_ratio_[0]:.2f}%)')
        plt.ylabel(f'PCA Component 2 (explained variance: {pca.explained_variance_ratio_[1]:.2f}%)')
        plt.title(
            f'First two components in 2D PCA Space')  # ,\nn real: {DH.labels.count(1)}, n fake: {DH.labels.count(0)}')

        plt.grid(True)
        plt.show()

        # plot the cumulative explained variance per number of components
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

if __name__ == "__main__":

    pca_path = Path('Dataset') / 'pca.pkl'

    try:
        if pca_path.stat().st_size == 0:
            make_pca(pca_path)
    except FileNotFoundError:
        make_pca(pca_path)




        Image.fromarray(pca_images[0]).show()