#!/bin/sh -x

MACHINE_TYPE=$1
INSTANCE_NAME="benchmark-instance-${MACHINE_TYPE}"
PROJECT="octoml-deelvin"

BENCHMARK_SAVE_PATH="$(pwd)/${MACHINE_TYPE}.csv"
MODELS_PATH="/home/agladyshev/Documents/classical-ml-models/models"
CLOUD_PYTHON_PROJECT_ROOT="~/msft-stuff/benchmark-performance-for-classical-ml-models"

gcloud compute instances create --project $PROJECT $INSTANCE_NAME \
  --no-service-account \
  --no-scopes \
  --zone=europe-west4-b \
  --machine-type=$MACHINE_TYPE \
  --image-project=ubuntu-os-cloud \
  --image=ubuntu-2004-focal-v20221202 \
  --boot-disk-type=pd-standard \
  --boot-disk-size=320GB \
  --maintenance-policy=TERMINATE

gcloud compute ssh --project $PROJECT $INSTANCE_NAME \
  --command="sudo apt-get update && \
             sudo apt-get install -y python3 python3-dev python3-setuptools python3-pip gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev"
gcloud compute ssh --project $PROJECT $INSTANCE_NAME \
  --command="git clone https://github.com/Deelvin/msft-stuff.git"
gcloud compute scp --project $PROJECT \
  --recurse ${MODELS_PATH} ${INSTANCE_NAME}:${CLOUD_PYTHON_PROJECT_ROOT}

gcloud compute ssh --project $PROJECT $INSTANCE_NAME \
  --command="cd ${CLOUD_PYTHON_PROJECT_ROOT} && \
             pip3 install -r requirements.txt"
gcloud compute ssh --project $PROJECT $INSTANCE_NAME \
  --command="cd ${CLOUD_PYTHON_PROJECT_ROOT} && \
             PYTHONPATH=./:$PYTHONPATH python3 scripts/benchmark.py"

gcloud compute scp --project $PROJECT \
  ${INSTANCE_NAME}:${CLOUD_PYTHON_PROJECT_ROOT}/benchmark_results.csv ${BENCHMARK_SAVE_PATH}

gcloud compute instances delete ${INSTANCE_NAME} --project $PROJECT
