#!/bin/bash

pushd ..
python receiver.py --use-gpu --fps 5 --signaling-host "192.168.0.201" --debug
popd