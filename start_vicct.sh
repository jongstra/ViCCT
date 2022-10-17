#!/bin/bash

#source venv/bin/activate && voila notebooks/make_image_prediction_voila.ipynb --no-browser &
#lt --subdomain amsterdamcrowdcounter --port 8866

source venv/bin/activate
# Add code to start notebook from terminal here.
onboardbase tunnels:create -p 8800 -s amsterdamcrowdcounter
