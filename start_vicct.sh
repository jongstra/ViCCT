#!/bin/bash

source venv/bin/activate && voila notebooks/make_image_prediction_voila.ipynb --no-browser &
lt --subdomain crowdcountingamsterdam --port 8866
