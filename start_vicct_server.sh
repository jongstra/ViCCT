#!/bin/bash

# Autostart this script at boot by running `crontab -e` and adding `@reboot PATH_TO_THIS_SCRIPT` to file.

#source venv/bin/activate && voila notebooks/make_image_prediction_voila.ipynb --no-browser &
#lt --subdomain amsterdamcrowdcounter --port 8866

source venv/bin/activate &&
python vicct_gradio.py &
sleep 10 && while true; do lt --subdomain amsterdamcrowdcounter --port 7860 && break; done
#sleep 10 && while true; do onboardbase tunnels:create -p 7860 -s amsterdamcrowdcounter && break; done