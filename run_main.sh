#!/bin/bash

source "/home/wpan1/Software/firedrake_wei/bin/activate"

#mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=240. --time_step=0.0025 --resolution=64 --nsave_data --nsave_visual --dump_freq=10 

# mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=2.0 --resolution=512 --time_step=0.0001 --dump_freq=10 --save_data --save_visual

#mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=8.275 --resolution=512 --time_step=0.00025 --dump_freq=4 --save_data --save_visual

#mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=0.067 --resolution=512 --time_step=0.00025 --dump_freq=4 --save_data --save_visual
#mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=1.5 --resolution=512 --time_step=0.00025 --dump_freq=4 --save_data --save_visual

#mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main_tqg_euler.py --time=15.5 --resolution=128 --time_step=0.00025 --dump_freq=4 --save_data --save_visual
mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main_tqg_random_bathymetry_process.py --time=20.5 --resolution=32 --time_step=0.0005 --dump_freq=20 --save_data --save_visual
# 0.0001  512
#mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/test_run_euler.py --time=1.5 --resolution=64 --time_step=0.00025 --dump_freq=4

deactivate

