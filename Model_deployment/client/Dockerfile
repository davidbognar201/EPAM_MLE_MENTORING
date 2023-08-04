FROM python:3.8
#FROM ubuntu

RUN apt-get update && \ 
            apt-get install -y python3 python3-pip && \
            apt-get -y install -qq --force-yes cron

COPY requirements.txt .
RUN pip3 install -r requirements.txt

#EXPOSE 8000