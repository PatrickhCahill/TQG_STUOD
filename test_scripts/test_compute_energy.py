from firedrake import *
import numpy as np

N=64 
mesh = PeriodicSquareMesh(N,N, 1.)
V    = FunctionSpace(mesh, "CG", 1)
W    = VectorFunctionSpace(mesh, "DG", 1)
x, y = SpatialCoordinate(mesh)

gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))
psi = Function(V).interpolate(sin(2*pi*x))
u = Function(W).project(gradperp(psi))
u_abs = assemble(abs(u))
print(u, u_abs)
print(type(u), type(u_abs))

# create grid
dx = 1/(N+1)
nx = np.linspace(0.0, 1.0, num=N+1)
energy_space = np.zeros((N+1, N+1))
grid = []
for i in nx:
    for j in nx:
        grid.append([i,j])
u_grid = np.asarray(u.at(grid, tolerance=1e-10))

total_energy = 0.5 * np.linalg.norm(u_grid, ord=2, axis=1)**2 *dx*dx
print(np.sum(total_energy), 0.5*norm(u)**2, 0.5*norm(u_abs, norm_type='L2')**2)
print(np.sum(0.5*np.linalg.norm(u.dat.data, ord=2, axis=1)**2 / u.dat.data.shape[0]))
print(u_grid.shape, u.dat.data.shape, u_abs.dat.data.shape)
