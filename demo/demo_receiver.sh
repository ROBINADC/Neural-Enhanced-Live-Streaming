#!/bin/bash

pushd ..
python receiver.py --debug \
    --record-sr-fn "sr.avi" --record-raw-fn "raw.avi" --fps 5 \
    --use-gpu --model-num-blocks 8 --model-num-features 8 \
    --signaling-host "wbserver.cs.usyd.edu.au" --signaling-port "13012" \
    --ice-config "ice.json" --ice-provider "xirsys"
popd