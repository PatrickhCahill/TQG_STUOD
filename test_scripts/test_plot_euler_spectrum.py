from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num_files = 1460 # 1 to 100 inclusive 
res = 64
fdataname = lambda n : "/home/wpan1/Data2/euler/data/pde_data_{}.npy".format(n)
ibase = 100

# time spectrum #################################
velocities = []
for i in range(1, num_files+1):
    vfile = np.load(fdataname(i))[123]
    # print(vfile.shape)
    velocities.append(vfile)
# print(velocities)    
data = 0.5*np.linalg.norm(velocities,ord=2,axis=1)**2
N = data.shape[-1]
dft = 2*np.fft.fft(data/N)
freq = np.fft.fftfreq(N)
mask = freq > 0
ps = np.abs(dft)**2
try:
    fig = plt.figure(1)
    plt.loglog(freq[mask],ps[mask])
    plt.savefig("./test_euler_spectrum_time.png")
except Exception as e:
    print("wat?")

# spatial spectrum ##############################
file_index = 100 

velocities = np.load(fdataname(file_index))

grid_values = 0.5*np.linalg.norm(velocities, ord=2, axis=1)**2 
grid_values = grid_values.reshape(res+1, res+1).T

xx = np.linspace(0,1, res+1)
yy = np.linspace(0,1, res+1)
[X,Y] = np.meshgrid(xx,yy)

fig = plt.figure(2)
ct = plt.contour(X,Y, grid_values)
plt.colorbar(ct)
plt.title("energy gridvalues")
plt.savefig("./test_euler_energy_gridvalues_{}.png".format(file_index))

fig = plt.figure(3)
ax = fig.add_subplot(111)
sampled_values = grid_values[:-1, :-1]
fourier_transform = 2*np.fft.fft2(sampled_values)
ps = np.abs(fourier_transform[:int(res/2), :int(res/2)])**2
print(ps.shape)
im = ax.imshow(np.log(ps), vmin=-20., vmax=0)
fig.colorbar(im)
plt.title("energy power spectrum (log scale)")
plt.savefig("./test_euler_energy_spectrum_{}.png".format(file_index))


# spatial spectrum animation #########################
if False:
    plt.style.use('seaborn-pastel')
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    im = ax.imshow( np.zeros( (int(res/2), int(res/2)) ), vmin=-20, vmax=0)
    ax.set_title("energy power spectrum (log scale)")
    ax.set(xlabel='x-axis frequencies', ylabel='y-axis frequencies')
    fig.colorbar(im)
    def init():
        im.set_data(np.zeros( (int(res/2), int(res/2)) ) )

    def animate(i):
        print("animate", i)
        velocities = np.load(fdataname(i+ibase))
        grid_values = 0.5*np.linalg.norm(velocities, ord=2, axis=1)**2 
        grid_values = grid_values.reshape(res+1, res+1).T
        sampled_values = grid_values[:-1, :-1]
        fourier_transform = 2*np.fft.fft2(sampled_values)
        ps = np.abs(fourier_transform[:int(res/2), :int(res/2)])**2
        # im = ax.imshow(np.log(ps))
        im.set_data(np.log(ps))
        return im

    # plt.savefig("./test_energy_spectrum.jpg")
    anim = FuncAnimation(fig, animate, init_func=init, frames=250, interval=50)
    anim.save('./test_euler_energy_spectrum.gif', writer='imagemagick')

# spatial spectrum radial animation ########################
if True:
    plt.style.use('seaborn-pastel')
    fig = plt.figure(5)
    ax = fig.add_subplot(111)
    im, = ax.loglog([1],[1])
    ax.set_title("energy power spectrum")
    ax.set(xlabel='log(k)', ylabel='log(E(k))')
    ax.set_xlim(.81, 50)
    ax.set_ylim(5e-8, 1)
    def init():
        # im.set_data(np.zeros( (int(res/2), int(res/2)) ) )
        im.set_data([1], [1])

    def animate(i):
        print("animate", i)
        velocities = np.load(fdataname(i+ibase))
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
        # im = ax.imshow(np.log(ps))
        im.set_data(x, y) # np.log(ps))
        return im

    # plt.savefig("./test_energy_spectrum.jpg")
    anim = FuncAnimation(fig, animate, init_func=init, frames=250, interval=50)
    anim.save('./test_euler_energy_spectrum_radial.gif', writer='imagemagick')

plt.close()


