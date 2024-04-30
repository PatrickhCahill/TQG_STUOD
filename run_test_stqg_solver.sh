#!/bin/bash

#source "/home/wpan1/Software/firedrake_wei/bin/activate"

# mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=240. --time_step=0.001 --resolution=64 --nsave_data --save_visual --dump_freq=100 
# mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=4.0 --resolution=512 --time_step=0.0001 --dump_freq=10 --save_data --save_visual
# mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=2.0 --resolution=512 --time_step=0.0001 --dump_freq=10 --save_data --save_visual

mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/test_stqg_solver.py --time=20. --time_step=0.001 --resolution=32 --nsave_data --save_visual --dump_freq=10
#mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/test_stqg_solver.py --time=1.5 --time_step=0.0002 --resolution=32 --nsave_data --save_visual --dump_freq=5
#deactivate

