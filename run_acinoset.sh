#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/acinoset"

docker build -q -f "$CURRENT_PATH"/docker/Dockerfile.acinoset.cui -t ${IMAGE_NAME} . && \
docker run -it --rm \
    -v "$CURRENT_PATH":/workdir \
    -v /data/naoya/AcinoSet:/data \
    -w /workdir \
    ${IMAGE_NAME} \
    /bin/bash
    # /bin/bash -c " \
    #     conda init bash && \
    #     conda activate acinoset && \
    #     jupyter lab --allow-root --NotebookApp.token='' --ip=* --no-browser \
    # "