#!/bin/bash
pushd ..

# use cpu 8 ~ 11
taskset 0f00 python server.py \
    --debug \
    --model-num-blocks 8 --model-num-features 8 \
    --signaling-host "wbserver.cs.usyd.edu.au" --signaling-port-sender "13011" --signaling-port-receiver "13012" \
    --ice-config "ice.json" --ice-provider "xirsys"

popd

echo "Finish."

