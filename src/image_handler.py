import random
import numpy as np
from PIL import Image

from pathlib import Path



class DatasetHandler:
    def __init__(self, dataset_path: str = 'Dataset', batch_size: int = 1000):
        '''

        :param dataset_path: the path (relative or absolute) to the directory containing the dataset.
        '''
        self.dataset_path: Path = Path(dataset_path)
        self.batch_size = batch_size   # the number of data points in a single batch
        self._batch_index = 0            # the index, organised per batch
        self._data = np.array([])  # a list for all the filenames
        self.labels = []
        self._find_images() # find all .jpg image files in the dataset
        self.label()
        self._datafile = None # nothing until it has been generated
        self._subset_file = None

        if not self._data:
            raise ValueError("no datapoints were found")

        self.training_data = np.array([])
        self.testing_data = np.array([])
        self.validation_data = np.array([])

    def _find_images(self, extension: str = 'jpg') -> None:
        '''
        Populate self.data with the names of the .jpg files for a given subset of the full dataset.

        :param extension: the extension of the files to be loaded: jpg by default.
        '''
        self._data = [file for file in self.dataset_path.rglob(f'*.{extension}')]#[:10000]

    def split(self, validation: bool = True) -> None:
        '''
        Divide the data into training, testing and validation sets

        :param validation: include a validation set or not
        :return:
        '''
        raise NotImplementedError

    def get_subset(self, size: int = 10000) ->list[str]:
        '''

        :param size: the number of image paths to return
        :return: a list of size paths to images, of which half are real and half are fake
        '''
        fake_images = [path for path in self._data if "Fake" in path]
        real_images = [path for path in self._data if "Real" in path]

        uneven = size % 2 == 0

        fake_sample = random.sample(fake_images, size // 2)
        real_sample = random.sample(real_images, size // 2 + (1 if uneven else 0))

        return fake_sample + real_sample

    def load_images(self, images: list[str], grayscale: bool = False, loading_batch_size: int = 1000) -> Path:
        '''

        :param images: a list of string paths to the image files to be loaded.
        :param grayscale:
        :param loading_batch_size:
        :return:
        '''

        size = len(images)
        self._subset_file = Path('Dataset') / (f'{size}_grayscale_images.npy' if grayscale else f'{size}_images.npy')

        batch_index = 0
        while batch_index < len(images):
            for index, path in enumerate(images[batch_index:batch_index+loading_batch_size]):
                with Image.open(path) as image:
                    if grayscale:
                        image = image.convert("L") # convert image to grayscale
                    self._subset_file[index] = np.asarray(image, dtype=np.float32).flatten().copy()
                batch_index += loading_batch_size

        print("finished loading subset into binary file")
        return self._subset_file

    def load_all_images(self, grayscale: bool = False, redo: bool = False, loading_batch_size: int = 1000) -> None:
        '''

        :param grayscale:
        :param redo:
        :param loading_batch_size:
        '''

        print('loading images...')

        path = Path('Dataset') / ('grayscale_images.npy' if grayscale else 'images.npy')
        channels = 1 if grayscale else 3

        self._datafile = np.memmap(path, dtype=np.float32, mode='w+', shape=(len(self._data), 256 * 256 * channels))
        print('file ready')

        if not np.all(self._datafile == 0) and not redo:
            print('images were already loaded')
            return None # file already exists

        batch_index = 0
        while batch_index < len(self._data):
            for index, path in enumerate(self._data[batch_index:batch_index+loading_batch_size]): # same indices in both containers
                with Image.open(path) as image:
                    if grayscale:
                        image = image.convert("L")  # convert image to grayscale
                    self._datafile[index] = np.asarray(image, dtype=np.float32).flatten().copy()
                self._datafile.flush()
                batch_index += loading_batch_size

        print('finished loading all images into binary file')
        return None

    def label(self):
        self.labels = np.asarray([1 if 'Real' in str(name) else 0 for name in self._data])
        print('real images: ', np.sum(self.labels == 1))
        print('fake images: ', np.sum(self.labels == 0))


    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        batch_start_index = self._batch_index * self.batch_size
        if batch_start_index >= len(self._datafile):
            raise StopIteration()

        print(f'returning batch {self._batch_index}')
        data = []
        for index in range(self.batch_size):
            try:
                image = self._datafile[batch_start_index + index]
                data.append(image)
            except IndexError:
                raise StopIteration()

        self._batch_index += 1
        return data


    def __getitem__(self, item: int) -> str:
        '''
        :param item: the index of the datapoint to be returned
        :return: the path to the file
        '''
        return self._data[item]

    def __len__(self) -> int:
        '''

        :return:
        '''
        return len(self._data)
