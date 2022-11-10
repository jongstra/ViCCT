#!/bin/bash

# Code to free up port 7860, if necessary.
#sudo kill -9 $(sudo lsof -t -i:7860)

# cd to the folder containing this file (the ViCCT project folder).
cd "$(dirname "$0")"

# Start virtual env.
source venv/bin/activate &&

# Start prediction website (locally).
python vicct_gradio.py &

# Start an auto-restarting localtunnel to make prediction website available on public web.
# Set an environment variable 'VICCT_DOMAIN' in local environment to override the default subdomain.
while true; do sleep 10 && lt --subdomain "${VICCT_DOMAIN:=crowdcounter}" --port 7860 && break; done
#sleep 10 && while true; do sleep 10 && onboardbase tunnels:create -p 7860 -s "${VICCT_DOMAIN:=crowdcounter}" && break; done