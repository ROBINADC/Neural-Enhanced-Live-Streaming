#!/bin/bash

# vlc ~/Epsilon/result/records/raw.avi --qt-start-minimized --no-embedded-video --width 640 --no-autoscale


# vlc --no-one-instance --video-x 160 --video-y 160 --width 320 ~/Epsilon/result/records/raw.avi &
# vlc --no-one-instance --video-x 160 --video-y 160 --width 320 ~/Epsilon/result/records/sr.avi &

vlc --no-one-instance ~/Epsilon/result/records/raw.avi &
vlc --no-one-instance ~/Epsilon/result/records/sr.avi &