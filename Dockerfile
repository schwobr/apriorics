FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-get update && apt-get upgrade -y

RUN apt-get install build-essential wget software-properties-common -y && apt-get update

RUN mkdir /apriorics

WORKDIR /apriorics

COPY . .

RUN pip install -U pip && pip install -r requirements.txt

#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin; \
#    mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600; \
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub; \
#    add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"; \
#    apt-get update; \

RUN HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]