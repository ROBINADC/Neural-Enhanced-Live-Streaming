setlocal
cd ..
python sender.py --debug ^
    --play-from "data/video/dana_480p_30fps.mp4" --framerate-degradation 6 ^
    --apply-psnr-filter ^
    --signaling-host "wbserver.cs.usyd.edu.au" --signaling-port "13011" ^
    --ice-config "ice.json" --ice-provider "xirsys"
endlocal