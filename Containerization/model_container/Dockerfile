FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3 python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /model

COPY ./data ./data
COPY ./python_files ./python_files
COPY ./trained_model ./trained_model

CMD ["python3", "python_files/testModel.py"]
