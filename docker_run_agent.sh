#!/bin/bash

. env.sh

CONTAINER_WORKDIR=/home/user/drl

CONDA_CMD="conda run -n $CONDA_ENV python $CONTAINER_WORKDIR/continuous_control.py run"

DOCKER_CMD="docker run \
-it -v $(pwd):$CONTAINER_WORKDIR \
$DOCKER_IMAGE_TAG \
bash -c \"${CONDA_CMD}\""

echo $DOCKER_CMD
eval $DOCKER_CMD