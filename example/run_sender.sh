#!/bin/bash

pushd ..
python sender.py --play-from "data/video/dana_480p_30fps.mp4" --framerate-degradation 6 \
    --apply-psnr-filter \
    --signaling-host "192.168.0.201" --debug
popd