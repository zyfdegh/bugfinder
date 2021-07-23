## catordog

## macOS

My running machine

* macOS 11.4
* python 3.9.5
* pip 21.1.1

* tensorflow 2.5.0
* numpy 1.19.5
* cuda (optional)

```sh
$ pip3 install tensorflow \
    matplotlib \
    keras \
```

## dataset
Kaggle Cats and Dogs Dataset

Download Page https://www.microsoft.com/en-us/download/details.aspx?id=54765

Unzip and move files to this project root like this
```sh
catdog-data
├── test/       # 2000 *.jpg
├── train
│   ├── cat/   # 10000 dog.*.jpg
│   └── dog/   # 10000 cat.*.jpg 
└── val
    ├── cat/   # 1000 cat.*.jpg
    └── dog/   # 1000 dog.*.jpg
```

## Reference
1. Image classification - TensorFlow https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
