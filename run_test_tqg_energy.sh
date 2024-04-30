
#!/bin/bash

source "/home/wpan1/Software/firedrake/bin/activate"

# mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/test_tqg_energy.py --time=20.0 --resolution=64 --time_step=0.001 --dump_freq=200 --nsave_data --save_visual
mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/test_tqg_energy.py --time=20.0 --resolution=64 --time_step=0.001 --dump_freq=20 --nsave_data --nsave_visual --alpha=16

# mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/test_for_Laura_doubly_periodic.py --time=10.0 --resolution=64 --time_step=0.0005 --dump_freq=100 --nsave_data --nsave_visual

deactivate

