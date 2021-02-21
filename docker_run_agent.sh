#!/bin/bash

. env.sh

CONTAINER_WORKDIR=/home/user/drl

CONTAINER_ID=$(docker run -d -v $(pwd):$CONTAINER_WORKDIR -ti $DOCKER_IMAGE_TAG)

CONDA_CMD="cd $CONTAINER_WORKDIR/drlnd-common && \
conda run -n $CONDA_ENV python setup.py install && \
conda run -n $CONDA_ENV python $CONTAINER_WORKDIR/multi_agent.py run"

DOCKER_CMD="docker exec -it $CONTAINER_ID bash -c \"${CONDA_CMD}\""

echo $DOCKER_CMD
eval $DOCKER_CMD

docker kill $CONTAINER_ID