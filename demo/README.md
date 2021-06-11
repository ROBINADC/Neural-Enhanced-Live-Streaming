# Demo

> This folder includes scripts used in presentation demo.

## Procedure
1. Run the server first
2. Then run the sender
3. Then run the reciever
4. Run display to show the videos

## Server

Server locates in the university.
It is a CPU server, and we used CPU 8 ~ 11 for this task.
TCP socket signaling facilities are set up in port 13011 and 13012
for sender and receiver respectively.

## Sender

> The sender script is written for Windows

Streamer sends a 30fps video from local file.
By indicating framerate degradation level 6,
the video is reduced to 5fps.

Additionally, specifying `--use-camera` in argument list
will stream from camera.
Please keep an eye on framerate degradation level.

## Receiver

Receiver obtains video and models from server.
Video with avi format can be played concurrent with recording.
Please make sure the fps and model settings are consistent.
Receiver enables GPU inferencing.

When the videos are recorded to local files,
run display program to have a look at them.
