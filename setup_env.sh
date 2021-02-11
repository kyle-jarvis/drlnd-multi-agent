#!/bin/bash

FILE=`readlink -f $0`
DIR=`dirname $FILE`

if [ "$#" -ne 1 ]
then
    echo "Please provide environment name as an argument"
    exit 1
fi

ENV_NAME="$1"

# Eval the conda bash hook

eval "$(conda shell.bash hook)"

# Setup the environment

if ! command -v conda &> /dev/null
then
    echo "conda could not be found"
    exit 1
fi

echo "Creating conda environment"

conda create --name $ENV_NAME python=3.6

if [ $? -eq 0 ]
then
    echo "Environment created successfully, continuing."
fi

conda activate $ENV_NAME

eval "pip install gym['box2d']"

DRLND_URL="https://github.com/udacity/deep-reinforcement-learning/"

echo "Installing requirements from $DRLND_URL"

pip install -e "git+$DRLND_URL#egg=unityagents&subdirectory=python"

pip install -r "$DIR/conf/extra_requirements.txt"

