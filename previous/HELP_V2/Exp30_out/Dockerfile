FROM tensorflow/tensorflow:2.0.0-gpu-py3

ENV SRC_DIR /src

COPY src $SRC_DIR
WORKDIR $SRC_DIR

RUN chmod +x ./train.sh ./inference.sh
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install -r requirements.txt
