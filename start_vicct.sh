#!/bin/bash

#source venv/bin/activate && voila notebooks/make_image_prediction_voila.ipynb --no-browser &
#lt --subdomain amsterdamcrowdcounter --port 8866

source venv/bin/activate
# Add code to start notebook from terminal here.
while true; do lt --subdomain amsterdamcrowdcounter --port 8800 && break; done
#while true; do onboardbase tunnels:create -p 8800 -s amsterdamcrowdcounter && break; done