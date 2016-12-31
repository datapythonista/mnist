# mnist: Python utilities to download and parse the MNIST dataset

## MNIST dataset

The MNIST database is available at http://yann.lecun.com/exdb/mnist/

The MNIST database is a dataset of handwritten digits. It has 60,000 training
samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each
containing a value 0 - 255 with its grayscale value.

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition
methods on real-world data while spending minimal efforts on preprocessing and formatting.

There are four files available, which contain separately train and test, and images and labels.

Thanks to Yann LeCun, Corinna Cortes, Christopher J.C. Burges

## Usage

mnist makes it easier to download and convert to Python objects the information
contained in the MNIST files, which use a custom format.

It supports Python 2.7 and Python >= 3.5.

To automatically download the train files, and display the first image in the
dataset, and print the number encoded in the image, you can simply use:

```
import scipy.misc
import mnist

train_x, rows, cols = mnist.mnist_images()
train_y = mnist.mnist_labels()

img = train_x[0,:].reshape((rows, cols))

print('Encoded number: %d' % train_y[0])
scipy.misc.toimage(scipy.misc.imresize(img * -1 + 256, 10.))
```

![](https://github.com/datapythonista/mnist/raw/master/img/img_5.png)

Test files can be downloaded in the same way, just by adding a parameter:

```
import mnist

test_x, rows, cols = mnist.mnist_images(test=True)
test_y = mnist.mnist_labels(test=True)

```

If you already have a file with the MNIST format, you can open it
with mnist too:

```
import mnist

train_x, rows, cols = mnist.mnist_images('train-images-idx3-ubyte.gz')
train_y = mnist.mnist_labels('train-labels-idx1-ubyte.gz')
```

By default, mnist returns the pixels and the labels as a
[numpy](http://www.numpy.org/) array. But it also can return
them as an [array](https://docs.python.org/3/library/array.html)
from Python standard lib, to avoid external dependencies. This module
is much more efficient than Python
[list](https://docs.python.org/3.6/library/stdtypes.html#list) type.

Note than when numpy is used, the images are returned one per row. While
when using array, a 1D dimension array is returned.

```
import mnist

train_x, rows, cols = mnist.mnist_images(use_numpy=False)
train_y = mnist.mnist_labels(use_numpy=False)
```
