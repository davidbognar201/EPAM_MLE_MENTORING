# As thhe homework description insisted I used a base Linux image as the base layer of the container
FROM ubuntu:latest

# Installing Python, Pip and Git on top of the base image, in line with the Docker best practises, 
# the 'apt-get install -y' command will ensure that the latest version of packages will be installed
RUN apt-get update && \ 
            apt-get install -y python3 python3-pip && \
            apt-get install -y vim git

# Copy the requirements.txt to the container and installing all the dependencies in it
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Setting environment variables
ENV USER_ID=${USER_ID:-1000}
ENV GROUP_ID=${USER_ID:-1000}

# Setting up the user's group and user id
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /model
RUN mkdir model_iterations 
