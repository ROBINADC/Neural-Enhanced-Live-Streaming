#!/bin/bash

pushd ..

python receiver.py --ice-config "ice.json" --ice-provider "xirsys" \
    --signaling-host "wbserver.cs.usyd.edu.au" --signaling-port "13012" \
    --model-num-blocks 8 --model-num-features 8 \
    --record-sr-fn "sr.avi" --record-raw-fn "raw.avi" --fps 5 \
    --debug

popd