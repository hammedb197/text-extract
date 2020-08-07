#Download base image ubuntu 18.04
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
ENV secret  "xxxx"


ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

RUN apt-get update && apt-get install -y \
    tar \
    wget \
    bash \
    rsync \
    gcc \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng-dev \
    libzmq3-dev \
    python3 \ 
    python3-dev \
    python3-pip \
    unzip \
    pkg-config \
    software-properties-common \
    graphviz



# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> ~/.bashrc

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



RUN pip3 install --upgrade pip
# install from PYPI using secret
RUN pip3 install spark-nlp==2.4.5
RUN pip3 install spark-ocr==1.5.0 --user --extra-index-url=https://pypi.johnsnowlabs.com/${secret} --upgrade
RUN pip3 install flask 


WORKDIR /app

COPY text_extract.py text_extract.py
ENTRYPOINT ["python3", "/app/text_extract.py"]

