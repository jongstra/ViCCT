#!/bin/bash

source venv/bin/activate && voila notebooks/make_image_prediction_voila.ipynb &
lt --subdomain crowdcountingamsterdam --port 8866
