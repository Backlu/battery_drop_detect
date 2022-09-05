FROM tensorflow/tensorflow:2.1.0-gpu-py3

WORKDIR /usr/src/app

COPY requirements.txt ./

ENV https_proxy http://10.110.15.60:8080/
ENV http_proxy http://10.110.15.60:8080
# --- 0602test
RUN apt-key adv --keyserver-options http-proxy=http://10.110.15.60:8080 --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
# ------------
RUN apt-get update -y && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN apt-get install -y libopencv-highgui-dev
RUN apt-get install -y libmysqlclient-dev
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y libtesseract-dev

RUN apt-get install -y iputils-ping
RUN apt-get update \
    &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata 

RUN python3 -m pip install --upgrade pip --proxy=http://10.110.15.60:8080
RUN pip install --no-cache-dir -r requirements.txt --proxy=http://10.110.15.60:8080 --default-timeout=1000
