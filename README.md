## bugfinder
Find 'cartoon bugs' from avatar images.

![Screenshot](https://raw.githubusercontent.com/zyfdegh/bugfinder/master/doc/bug.png)

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

## Docker
```sh
docker build -t zyfdedh/bugfinder .
docker run zyfdedh/bugfinder
```

eec4912fe3a00eef6f5fe341e4e60a90.jpg: bugs, confidence:99.2
f18688893ce41f6da475785f6fd567e3.jpg: normal, confidence:100.0
f4335898c073b18a3d1d3a85063a89f4.jpg: normal, confidence:100.0
f7800644f66ce0532935dc4c0b514b75.jpg: normal, confidence:100.0
f9048664c12b94ffdd4ca54aff57ac70.jpg: normal, confidence:100.0
ee4fb207a66eeaeb2a5b5e9a789a5d6c.jpg: bugs, confidence:99.4
f6585975a6dcdd9bc54c356e988becc3.jpg: normal, confidence:100.0
f4219696c995e94bd16357a4437a325d.jpg: normal, confidence:100.0
f4891441f17961604f271156e0eaadcf.jpg: normal, confidence:100.0
f4850204df97a4186e88ad0208933f61.jpg: normal, confidence:100.0

## dataset
Kaggle Cats and Dogs Dataset

Download Page https://www.microsoft.com/en-us/download/details.aspx?id=54765

Unzip and move files to this project root like this
```sh
imgs
├── test/
├── train
│   ├── bugs/
│   └── normal/
└── val
    ├── bugs/
    └── normal/
```

## Reference
1. Image classification - TensorFlow https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
