
"""Functions for downloading and reading MNIST data."""
import gzip
import os
import urllib
import numpy


import cv2

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, images, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in range(784)]
            return [fake_image for _ in range(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    VALIDATION_SIZE = 100
    list = os.listdir("./edge")
    aaa=[]
    for name in list:
        img = cv2.imread("./edge/"+name)
        arr=numpy.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        data = arr.reshape(28, 28, 1)
        data=numpy.nan_to_num(data)
        aaa.append(data)
    size=len(aaa)
    train_images = numpy.asarray(aaa)
    train_images.reshape(size,28,28,1)
    # print(train_images)
    validation_images = train_images[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    data_sets.train = DataSet(train_images)
    data_sets.validation = DataSet(validation_images)
    return data_sets