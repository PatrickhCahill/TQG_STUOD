from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from data_assimilation.data_assimilation_utilities import data_assimilation_workspace
from firedrake_utility import TorusMeshHierarchy
from tqg.solver import TQGSolver
from tqg.example2 import TQGExampleTwo as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm
ensemble_comm = ensemble

nx = 512
T= 1.
dt = 0.0001

df_wspace = data_assimilation_workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")
mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=comm).get_fine_mesh()
params = Example2params(T, dt, mesh, bc='x', alpha=None)

solver = TQGSolver(params)
subdir = "PDESolution-random-bathymetry"
subdir_original = "PDESolution-no-bathymetry-perturbation"

file_index = 670 

def pde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("pde_data_{}".format(file_index), sub_dir) 

data_name_random_bathymetry = pde_data_fname(df_wspace, file_index, subdir)
data_name_original = pde_data_fname(df_wspace, file_index, subdir_original)

velocities_random_bathymetry = solver.get_velocity_grid_data(data_name_random_bathymetry , nx)
velocities_original = solver.get_velocity_grid_data(data_name_original, nx)

res = nx

# spatial spectrum ##############################
print("spatial spectrum")
#velocities = np.load(fdataname(file_index) )

def compute_spectrum_radial(velocities, res=512):
    grid_values = 0.5*np.linalg.norm(velocities, ord=2, axis=1)**2 
    grid_values = grid_values.reshape(res+1, res+1).T

    sampled_values = grid_values[:-1, :-1]
    fourier_transform = 2*np.fft.fft2(sampled_values)
    ps = np.abs(fourier_transform[:int(res/2), :int(res/2)])**2

    k_val_dic = {}
    for i in range(ps.shape[0]):
        for j in range(ps.shape[1]):
            ksquared = i**2 + j**2
            if (ksquared in k_val_dic):
                k_val_dic[ksquared].append(ps[i,j])
            else:
                k_val_dic[ksquared] = [ps[i,j]]
    k_val_dic_sorted = sorted(k_val_dic.items(), key = lambda item: item[0])

    x, y = [], []
    for key, val in k_val_dic_sorted:
        x.append(np.sqrt(key))
        y.append(np.mean(val))
    return k_val_dic_sorted, x, y


k_val_dic_sorted, x, y = compute_spectrum_radial(velocities_random_bathymetry)
k_val_dic_sorted_o, x_o, y_o = compute_spectrum_radial(velocities_original)

fig = plt.figure(5)
ax = fig.add_subplot(111)
# im, = ax.loglog([1],[1])
ax.set_title("energy power spectrum")

from scipy.ndimage import gaussian_filter1d
y_3 = gaussian_filter1d(y,sigma=1)    
y_4 = gaussian_filter1d(y_o, sigma=1)

np.save("./y_3", np.asarray(y_3))
np.save("./y_4", np.asarray(y_4))
np.save("./x", np.asarray(x))
np.save("./x_o", np.asarray(x_o))
#ax.scatter(np.log(x), np.log(y_3))
# ax.loglog(x, y_3)
ax.scatter(x, y_3,s=0.02, c='blue', alpha=1)
ax.scatter(x_o, y_4, s=0.02, c='orange', alpha=1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(xlabel='log(k)', ylabel='log(E(k))')
#ax.set_ylim(5e-6, 1e12)
ax.set_xlim(.81, 5e2)
plt.savefig("./test_fine_res_energy_spectrum_radial_{}.png".format(file_index))

