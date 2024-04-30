#!/bin/bash

#source "/home/wpan1/Software/firedrake/bin/activate"

python data_assimilation/generate_observation_error_covariance.py
python data_assimilation/generate_observation_data.py
python run_uncertainty_quantification.py
python test_scripts/test_satellite_obs_visual.py

#deactivate
