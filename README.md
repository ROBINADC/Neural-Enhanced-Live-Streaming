# Neural-Enhanced Super-Resolution for Real-Time Video Streaming

## Introduction

The proposed framework, Epsilon, is a real-time video streaming system 
that utilizes server's and viewer's computational resources to enhance live video quality.
Epsilon concerns two communicating peers,
video sender and video receiver, which are interconnected by a streaming server.
The framework is built on top of `aiortc`.

## System Design

In our scenario, a streamer streams a low-resolution (LR) video to the server.
At the same time, the streamer will sample training patches, which are sent to the server along with the video.
Upon receiving the LR video sent from streamer, the server relays the video to the viewer. 
The server also maintains a super-resolution (SR) model for the video, and uses patches to train it.
After each training epoch, the trained model is delivered to the viewer.
Once the viewer receives an SR model, 
it applies it to the ongoing LR video in a per-frame super-resolution manner, generating a high-resolution (HR) video.

<div align="center">
    <img src="./assets/design.svg" width="60%" alt="System Design" />
</div>

## Files

### Core

```
.
├── dataset.py      dataset objects used for neural training
├── media.py        a copy of aiortc.contrib.media with modifications on some classes and functions
├── misc.py         utilities
├── model.py        the Super-Resolution neural network and its components
├── receiver.py     implementation of video viewer
├── sender.py       implementation of video streamer
└── server.py       live-streaming server augmented with online neural training
```

### Aside

```
.
├── analysis        scripts for analysis and visualization
├── assets          images in README
├── data
│   └── video       contains a sample video  
├── demo            scripts used in the university demo
├── example         scripts with minimal settings to run the system [USE THIS FOR QUICKSTART]
├── ice.json        configuration for STUN and TURN servers (not a necessity)
├── offline.py      an offline training example
└── result          result generated at runtime
    ├── logs        runtime log files
    └── records     recorded raw and super-resolved video
```

## Dependencies

The system runs on Python 3.8 with following third-party libraries:
```
aiortc==1.2.0
numpy==1.19.2
opencv==4.0.1
pytorch==1.7.1
torchvision==0.8.2
```
We also use `matplotlib` and `pandas` in analysis part, but they are not essential for the system.

## Usage (LAN)

- Server - `server.py`
- Streamer - `sender.py`
- Viewer - `receiver.py`

Whenever possible, run three programs in three seperate computers.
When only two computers are available, run streamer and viewer on the same device.

### Step 1 - start the server

To start the server, you need to specify WebRTC's signaling server,
and assign two ports for connecting sender and receiver.

**Frequently-used arguments**
```
--model-num-blocks [NUM] : the number of residual block of the SR model
--model-num-features [NUM] : the number of channels for convolutional filters
--use-gpu
--signaling-host [ADDRESS_TO_SIGNALING_SERVER] : signaling server address
--signaling-port-sender [PORT]
--signaling-port-receiver [PORT] 
--debug : detailed logging
```

### Step 2 - start the sender

The streamer can use local file or webcam as media source.

**Frequently-used arguments**
```
--play-from [VIDEO_FILE] : play from local file 
--framerate-degradation [LEVEL] : reduce framerate to accommodate weak machines
--apply-psnr-filter : use PSNR filter
--signaling-host [ADDRESS_TO_SIGNALING_SERVER] : signaling server address
--signaling-port [PORT]
--debug : detailed logging
```

### Step 3 - start the receiver

To run the receiver, make sure the framerate is correctly specified.

**Frequently-used arguments**
```
--record-sr-fn [FILE] : the file name of the super-resolved video
--record-raw-fn [FILE] : the file name of the raw video
--fps [FPS] : recording framerate (must identical to the sending framerate)
--model-num-blocks [NUM] : the number of residual block of the SR model
--model-num-features [NUM] : the number of channels for convolutional filters
--use-gpu
--signaling-host [ADDRESS_TO_SIGNALING_SERVER] : signaling server address
--signaling-port [PORT]
--debug : detailed logging
```

### Termination 

Use keyboard interrupt to stop any program.

> A scripting example with executable settings is provided in the directory `example`
> 
> A demo-oriented example is provided in the directory `demo`

## WAN and ICE

A straight up connection across WAN between two peers might not work for many reasons.

- Intermediate firewalls may block such connections.
- Peers might not have public IP address to distinguish themselves.
- Routers might not allow direct connection to certain peers.

Interactive Connectivity Establishment (ICE) is a framework for establishing peer connections despite the above limitations.
ICE uses STUN and/or TURN internally.
  
In the CLI, specify relevant arguments to explicitly use ICE in the system.
A sample ice configuration file is provided in `ice.json`.
```
--ice-config [ICE_CONFIGURATION_FILE] : JSON file for ICE server configurations
--ice-provider [KEY] : a top-level key in the JSON file (such as "google" or "xirsys")
```

## Reference

**LiveNAS**.
Kim, J., Jung, Y., Yeo, H., Ye, J., & Han, D. (2020). 
Neural-enhanced live streaming: Improving live video ingest via online learning. 
Proceedings of the Annual Conference of the ACM Special Interest Group on Data Communication 
on the Applications, Technologies, Architectures, and Protocols for Computer Communication, 107–125. 
https://doi.org/10.1145/3387514.3405856

**NAS**.
Yeo, H., Jung, Y., Kim, J., Shin, J., & Han, D. (2018). 
Neural adaptive content-aware internet video delivery. 
13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), 645–661. 
https://www.usenix.org/conference/osdi18/presentation/yeo

**MDSR**.
Lim, B., Son, S., Kim, H., Nah, S., & Lee, K. M. (2017). 
Enhanced deep residual networks for single image super-resolution. 
2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 1132–1140. 
https://doi.org/10.1109/CVPRW.2017.151

**ICE**.
https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Protocols

**Provided video**.
https://www.youtube.com/watch?v=4AtOU0dDXv8