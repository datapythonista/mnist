# mnist: Python utilities to download and parse the MNIST dataset

## MNIST dataset

The MNIST database is available at http://yann.lecun.com/exdb/mnist/

The MNIST database is a dataset of handwritten digits. It has 60,000 training
samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each
containing a value 0 - 255 with its grayscale value.

![](https://github.com/datapythonista/mnist/raw/master/img/samples.png)

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition
methods on real-world data while spending minimal efforts on preprocessing and formatting.

There are four files available, which contain separately train and test, and images and labels.

Thanks to Yann LeCun, Corinna Cortes, Christopher J.C. Burges.

## Usage

mnist makes it easier to download and parse MNIST files.

To automatically download the train files, and display the first image in the
dataset, you can simply use:

```python
import mnist
import scipy.misc

images = mnist.train_images()
scipy.misc.toimage(scipy.misc.imresize(images[0,:,:] * -1 + 256, 10.))
```

![](https://github.com/datapythonista/mnist/raw/master/img/img_5.png)

Test files and labels can be downloaded in a similar way:

```python
import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()
```

The dataset is downloaded and cached in your temporary directory, so, calling
the functions again, is much faster and doesn't hit the server.

Images are returned as a 3D numpy array (samples * rows * columns). To train
machine learning models, usually a 2D array is used (samples * features). To
get it, simply use:

```python
import mnist

train_images = mnist.train_images()
x = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
```

Both the url where the files can be found, and the temporary directory where
they will be cached locally can be modified in the next way:
```python
import mnist

mnist.datasets_url = 'http://url-to-the/datasets'

# temporary_dir is a function, so it can be dinamically created
# like Python stdlib `tempfile.gettempdir` (which is the default)
mnist.temporary_dir = lambda: '/tmp/mnist'

train_images = mnist.train_images()
```

It supports Python 2.7 and Python >= 3.5.
