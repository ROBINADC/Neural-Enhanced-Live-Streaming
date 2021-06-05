#!/bin/bash

# remote server
# the script should be executed in correct virtual environment (aio) that contains correct python packages

pushd ..

# use cpu 8 ~ 11
taskset 0f00 python server.py \
    --ice-config "ice.json" --ice-provider "xirsys" \
    --signaling-host "wbserver.cs.usyd.edu.au" --signaling-port-sender "13011" --signaling-port-receiver "13012" \
    --model-num-blocks 8 --model-num-features 8 \
    --debug

# taskset f000 python server.py \
#     --ice-config "ice.json" --ice-provider "xirsys" \
#     --signaling-host "wbserver.cs.usyd.edu.au" --signaling-port-sender "13015" --signaling-port-receiver "13016" \
#     --debug &

popd

echo "Finish."

