FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \ 
            apt-get install -y python3 python3-pip && \
            apt-get install -y vim git && \
            apt-get install -y awscli

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ARG USER_ID
ARG GROUP_ID
ARG AWS_KEY
ARG AWS_SECRET_KEY

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /data_gov
COPY --chown=user:user . /data_gov

