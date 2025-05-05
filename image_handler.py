
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

    def load_images(self, grayscale: bool = False, redo: bool = False) :
        '''

        :param grayscale:
        :param redo:

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
            for index, path in enumerate(self._data[batch_index:batch_index+1000]): # same indices in both containers
                with Image.open(path) as image:
                    if grayscale:
                        image = image.convert("L")  # convert image to grayscale
                    self._datafile[index] = np.asarray(image, dtype=np.float32).flatten().copy()
                self._datafile.flush()
                batch_index += 1000

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