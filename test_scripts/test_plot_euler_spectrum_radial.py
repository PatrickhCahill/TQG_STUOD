from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# time spectrum #################################
num_files = 1460 # 1 to 100 inclusive 
res = 64
fdataname = lambda n : "/home/wpan1/Data2/euler/data/pde_data_{}.npy".format(n)

# velocities = []
# for i in range(1, num_files+1):
#     vfile = np.load(fdataname(i))[123]
#     # print(vfile.shape)
#     velocities.append(vfile)
# # print(velocities)    
# data = 0.5*np.linalg.norm(velocities,ord=2,axis=1)**2
# N = data.shape[-1]
# dft = 2*np.fft.fft(data/N)
# freq = np.fft.fftfreq(N)
# mask = freq > 0
# ps = np.abs(dft)**2
# try:
#     fig = plt.figure(1)
#     plt.loglog(freq[mask],ps[mask])
#     plt.savefig("./test_euler_spectrum_time.png")
# except Exception as e:
#     print("wat?")

# spatial spectrum ##############################
file_index = 100


velocities = np.load(fdataname(file_index))

grid_values = 0.5*np.linalg.norm(velocities, ord=2, axis=1)**2 
grid_values = grid_values.reshape(res+1, res+1).T

xx = np.linspace(0,1, res+1)
yy = np.linspace(0,1, res+1)
[X,Y] = np.meshgrid(xx,yy)

fig = plt.figure(1)
ax = fig.add_subplot(111)
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
ax.loglog(x, y)
ax.set_ylim(5e-8,1)
ax.set_xlabel("log(k)")
ax.set_ylabel("log(E(k))")
plt.title("energy power spectrum (t step {})".format(file_index))
plt.savefig("./test_euler_energy_spectrum_radial_{}.png".format(file_index))


