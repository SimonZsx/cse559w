import numpy as np
import os
import urllib
import gzip
import struct

import matplotlib.pyplot as plt

def download_data(url, force_download=True):
    """Download data file to disk and returns filename."""
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_url, image_url):
    """Download and deserialize raw data to numpy ndarray. Return (label, image) tuple."""
    # the original files are gzip-compressed with a particular serialization format
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

path='http://yann.lecun.com/exdb/mnist/'

# label, image tuple
(train_lbl, train_img) = read_data(
    path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
(test_lbl, test_img) = read_data(
    path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')

# check raw data

# type(train_lbl)
# numpy.ndarray
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html

# train_lbl.shape
# (60000,)
# train_lbl.dtype
# dtype('int8')

# train_img.shape
# (60000, 28, 28)
# train_img.dtype
# dtype('uint8')


num_inst = 10
for i in range(num_inst):
    plt.subplot(1,num_inst,i+1)
    # train_img[i] has shape (28, 28)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:num_inst],))


