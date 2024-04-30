from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

res = 50
mesh = PeriodicUnitSquareMesh(res, res)
x,y = SpatialCoordinate(mesh)
 
V = FunctionSpace(mesh, "DG", 1)
u_exact = Function(V)
u_exact.interpolate(sin(2*pi*x)*sin(2*4*pi*y) + sin(2*pi*8*x))

# plot u_exact
fig = plt.figure(1)
fig, axes = plt.subplots(2, 1)
plt.subplots_adjust(hspace=1)

contours = tricontour(u_exact, axes=axes[0])
axes[0].set_title('fdrake')

# obtain grid
xx = np.linspace(0, 1, res+1)
yy = np.linspace(0, 1, res+1)

mygrid = []
for i in range(len(xx)):
    for j in range(len(yy)):
        mygrid.append([xx[i], yy[j]])
mygrid = np.asarray(mygrid) # now i have a list of grid values

grid_values = np.asarray(u_exact.at(mygrid, tolerance=1e-10)) 
grid_values = grid_values.reshape(res+1, res+1).T

# meshgrid for contour plot
[X,Y] = np.meshgrid(xx, yy)
axes[1].contour(X,Y,grid_values)
axes[1].set_title('meshgrid_values')

plt.savefig("./test_fdrake_grid_values.png")
plt.close()

fig = plt.figure(2)
# Fourier analysis working with grid_values
sampled_values = grid_values[:-1,:-1] # need to get rid off the last point for periodicity
fourier_transform = 2*np.fft.fft2(sampled_values)
ps = np.abs(fourier_transform[:int(res/2),:int(res/2)])**2

plt.title("power spectrum")
plt.imshow(ps)
plt.savefig("./test_fdrake_spectrum.png")

plt.close()
