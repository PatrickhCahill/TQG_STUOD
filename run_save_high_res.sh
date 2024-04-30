#!/bin/bash

source "/home/wpan1/Software/firedrake/bin/activate"

mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/save_high_res_psi_grid_value_time_series.py
# mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/save_high_res_velocity_grid_values.py

deactivate

