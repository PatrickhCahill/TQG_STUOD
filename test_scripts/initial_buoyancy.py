"""
1 May 2020

This script uses a double Poisson formulation to get a particular solution for the initial_buoyancy
"""

from firedrake import *
import math

L = 1.
nx = 64

msh = PeriodicSquareMesh(nx,nx,L)

Vcg = FunctionSpace(msh, "CG", 1)

h1 = Function(Vcg)
w0 = Function(Vcg)
x= SpatialCoordinate(Vcg.mesh())
nullspace = VectorSpaceBasis(constant=True)

# h1.interpolate(2.0*cos(2.0*math.pi*x[0]/L))   # = 2 h_1 in the notes
h1.interpolate( (2.0+8.0 * math.pi**2 / L**2)* cos(2.0*math.pi*x[0]/L) )

w = TrialFunction(Vcg)
phi = TestFunction(Vcg)

# --- Elliptic equation ---
# Build the weak form for the inversion
# stream function is 0 on the boundary - top and bottom, left and right
Aw = (inner(grad(w), grad(phi))) * dx
Lw = -h1 * phi * dx 
#- (inner(grad(h1), grad(phi))) * dx

w_problem = LinearVariationalProblem(Aw, Lw, w0)
w_solver = LinearVariationalSolver(w_problem) #, nullspace=nullspace)
# , solver_parameters={
#     'ksp_type': 'preonly',
#     'pc_type': 'lu',
#     'pc_sub_type': 'ilu'
#     # 'ksp_type':'cg',
#     # 'pc_type':'sor'
# })

b0 = Function(Vcg)
u = TrialFunction(Vcg)
phi2 = TestFunction(Vcg)

Ab = (inner(grad(u), grad(phi2))) * dx
Lb = -w0 * phi2 * dx

b_problem = LinearVariationalProblem(Ab, Lb, b0)
b_solver = LinearVariationalSolver(b_problem, nullspace=nullspace)
# , solver_parameters={
#     'ksp_type': 'preonly',
#     'pc_type': 'lu',
#     'pc_sub_type': 'ilu'
#     # 'ksp_type':'cg',
#     # 'pc_type':'sor'
# })

# solve and output
w_solver.solve()
b_solver.solve()

#

# try:
#     import matplotlib.pyplot as plt
# except:
#     warning("Matplotlib not imported")

# try:
#     fig, axes = plt.subplots()
#     colors = tripcolor(b0, axes=axes)
#     fig.colorbar(colors)
# except Exception as e:
#     warning("Cannot plot figure. Error msg '%s'" % e)

# try:
#     plt.savefig('initial_buoyance.png',dpi=300)
#     # plt.show()
# except Exception as e:
#     warning("Cannot show figure. Error msg '%s'" % e)
b0.rename("InitialBuoyancy")
file = File('initial_buoyancy.pvd')
file.write(b0)
