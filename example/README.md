# Example

A rather easy-to-configure example to run the program in LAN.

In this example, we use two devices to demonstrate how Epsilon works.
The server is hosted on a Windows device, while the sender and receiver reside on another Linux device.
Both devices are equipped with GPUs.

If you have other device or OS configuration, please refer to these scripts to write your own scripts.

## Procedure

1. In all files replace the value of `--signaling-host` with the IP address of the device running the server program
2. Run `run_server.bat` in a Windows device (with GPU)
3. Run `run_sender.sh` in a Linux device
4. Run `run_receiver.sh` in the same Linux device (with GPU)
