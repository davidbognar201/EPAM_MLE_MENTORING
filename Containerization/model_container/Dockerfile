# As the homework description insisted I used a base Linux image as the base layer of the container
FROM ubuntu:latest

# Installing Python, Pip and Git on top of the base image
RUN apt-get update && \ 
            apt-get install -y python3 python3-pip && \
            apt-get install -y vim git

# Copy the requirements.txt to the container and installing all the dependencies in it
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Setting the working directory and copying all the required files such as the train and test data and the .py scripts
WORKDIR /model
COPY ./data ./data
COPY ./python_files ./python_files
COPY ./trained_model ./trained_model
