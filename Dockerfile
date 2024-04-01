FROM ubuntu:18.04
RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev  libgl1-mesa-glx tesseract-ocr
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . /app
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]