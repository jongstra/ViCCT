#!/bin/bash

# Autostart this script at boot by running `crontab -e` and adding `@reboot PATH_TO_THIS_SCRIPT` to file.

# Code to free up port 7860, if necessary.
#sudo kill -9 $(sudo lsof -t -i:7860) &&

# Start virtual env.
source venv/bin/activate &&

# Start prediction website (locally).
python vicct_gradio.py &

# Start an auto-restarting localtunnel to make prediction website available on public web.
while true; do sleep 10 && lt --subdomain amsterdamcrowdcounter --port 7860 && break; done
#sleep 10 && while true; do sleep 10 && onboardbase tunnels:create -p 7860 -s amsterdamcrowdcounter && break; done