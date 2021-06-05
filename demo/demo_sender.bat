setlocal
cd ..
python sender.py --ice-config "ice.json" --ice-provider "google" ^
    --signaling-host "wbserver.cs.usyd.edu.au" --signaling-port "13011" ^
    --play-from "data/video/dana_480p_30fps.mp4" --framerate-degradation 6 --debug
endlocal