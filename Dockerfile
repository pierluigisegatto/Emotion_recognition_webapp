# this is an official Python runtime, used as the parent image
FROM python:3.8-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc make && \
    apt-get -y install libsndfile-dev supervisor &&\
    apt clean && rm -rf /var/lib/apt/lists/*


# Install unzip and curl for dataset
RUN apt-get update && apt install -y unzip && \
    apt install -y curl

# create log folder for supervisor
RUN mkdir -p /var/log/supervisor
# copy configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# create the root of the project
RUN mkdir Visium_project

# set the working directory in the container to /Visium project
WORKDIR /Visium_project

# copy all local files to project folder
COPY . .

# execute everyone's favorite pip command, pip install -r
RUN pip install pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# cd into data
WORKDIR /Visium_project/proj/app/data

# Download the emo-db dataset and remove all unnecessary files
RUN curl -o emodb.zip http://emodb.bilderbar.info/download/download.zip && \
    unzip emodb.zip && mv wav ../ && mv data_reader.py ../ && rm -r * && \
    mv ../wav . && mv ../data_reader.py .

# cd into app home
WORKDIR /Visium_project/proj/app

# unblock port 5000 for the Flask app to run on
EXPOSE 5000
EXPOSE 8888

# execute the Flask app
CMD ["/usr/bin/supervisord"]
