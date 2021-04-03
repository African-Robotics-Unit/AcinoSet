#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/acinoset"

docker build -q -f "$CURRENT_PATH"/docker/Dockerfile.acinoset -t ${IMAGE_NAME} . && \
docker run -it --rm \
    --gpus device=0 \
    -v "$CURRENT_PATH":/workdir \
    -v /disk2/naoya/AcinoSet:/data \
    -w /workdir \
    -p 8888:8888 \
    ${IMAGE_NAME} \
    /bin/bash
    # /bin/bash -c " \
    #     conda init bash && \
    #     conda activate acinoset && \
    #     jupyter lab --allow-root --NotebookApp.token='' --ip=* --no-browser \
    # "