#!/bin/bash

FILE=`readlink -f $0`
DIR=`dirname $FILE`
UNITY_ENV_DIR="unity_environments"
UNITY_ENV_URL="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3"


if [ ! -d $UNITY_ENV_DIR ]
then
    mkdir $UNITY_ENV_DIR
fi

cd $UNITY_ENV_DIR

get_unity_env () {
echo "Looking for ./$UNITY_ENV_DIR/$1"
if [ ! -d "$DIR/$UNITY_ENV_DIR/$1" ]
then
    echo "Downloading and unzipping environment: $1"
    wget --directory-prefix $(basename $(dirname $1)) $UNITY_ENV_URL/$1.zip
    unzip "$1.zip" -d $(dirname $1)
else
    echo "$DIR/$UNITY_ENV_DIR/$1 exists, skipping"
fi
}

declare -a envs=("Tennis/Tennis_Linux" "Soccer/Soccer_Linux")
for ENV in "${envs[@]}"
do
    get_unity_env $ENV
done
