import mxnet as mx
import numpy as np
import os
import urllib
import gzip
import struct
import matplotlib.pyplot as plt

from mxnet import nd, autograd, gluon
mx.random.seed(1)

def download_data(url, force_download=True):
    """Download data file to disk and returns filename."""
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_url, image_url):
    """Download and deserialize raw data to numpy ndarray. Return (label, image) tuple."""
    # the original files are gzip-compressed with a particular serialization format
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

path='./'

# label, image tuple
(train_lbl, train_img) = read_data(
    path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
(test_lbl, test_img) = read_data(
    path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')
def to4d(img):
    """Reshape img to 4d tensor and normalize pixel values to [0, 1]."""
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

batch_size = 100
train_iter = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(to4d(train_img), train_lbl), batch_size, shuffle=True)
test_iter = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(to4d(test_img), test_lbl), batch_size)

# construct cnn
num_fc = 512
num_outputs = 10
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_fc, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    
print(net)
model_ctx = mx.cpu()
#model_ctx = mx.gpu()

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=model_ctx)

def evaluate_accuracy(data_iterator, net):
    """Make predictions for the dataset and evaluate average accuracy."""
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        # ==== note the difference in raw data input shape ====
        # use 4d tensor (batch_size, 1, 28, 28)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

epochs = 10
smoothing_constant = .01
num_examples = 60000

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_iter):
        # ==== note the difference in raw data input shape ====
        # use 4d tensor (batch_size, 1, 28, 28)
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_iter, net)
    train_accuracy = evaluate_accuracy(train_iter, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))


# show test image
plt.imshow(test_img[0], cmap='Greys_r')
plt.axis('off')
plt.show()

# make prediction
output = net(test_img[0:1])
print(""+str(np.asscalar(nd.argmax(output, axis=1).asnumpy().astype(np.int8))))