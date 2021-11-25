FROM tensorflow/tensorflow:2.7.0
WORKDIR /root

ADD . $WORKDIR

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install matplotlib

CMD ["python3", "bugfinder.py"]