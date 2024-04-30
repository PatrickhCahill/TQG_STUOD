
#!/bin/bash

source "/home/wpan1/Software/firedrake/bin/activate"

# mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=240. --time_step=0.001 --resolution=64 --nsave_data --save_visual --dump_freq=100 
mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/main_tqg_analysis_paper.py --time=2.5 --resolution=256 --time_step=0.0005 --dump_freq=40 --save_data --save_visual --alpha=256
# mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/main.py --time=2.0 --resolution=512 --time_step=0.0001 --dump_freq=10 --save_data --save_visual

# mpiexec -n 1 python /home/wpan1/Development/PythonProjects/ThermalQG/test_stqg_solver.py --time=4. --time_step=0.0001 --resolution=64 --save_data --save_visual --dump_freq=100
deactivate

