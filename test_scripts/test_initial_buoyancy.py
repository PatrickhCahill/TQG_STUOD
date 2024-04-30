"""
Test firedrake solver for initial buoyancy
"""

from firedrake import *
import math

L = 1.
nx = 512

msh = PeriodicSquareMesh(nx,nx,L)

Vcg = FunctionSpace(msh, "CG", 1)

h1 = Function(Vcg)
w0 = Function(Vcg)  # solution to the first Poisson problem
x= SpatialCoordinate(Vcg.mesh())
nullspace = VectorSpaceBasis(constant=True)

h1.interpolate(cos(2.0*math.pi*x[0]/L))   # = 2 h_1 in the notes

w = TrialFunction(Vcg)
phi = TestFunction(Vcg)

# --- Elliptic equation ---
# Build the weak form for the inversion
# stream function is 0 on the boundary - top and bottom, left and right
Aw = (inner(grad(w), grad(phi))) * dx
Lw = -2.0 *h1 * phi * dx - 2.0*(inner(grad(h1), grad(phi))) * dx


w_problem = LinearVariationalProblem(Aw, Lw, w0)
w_solver = LinearVariationalSolver(w_problem, nullspace=nullspace, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_sub_type': 'ilu'
    # 'ksp_type':'cg',
    # 'pc_type':'sor'
})


# --- Elliptic equation ----
b0 = Function(Vcg)     # solution to the second Poisson problem
g = Function(Vcg)
g.interpolate( (2.0+8.0 * math.pi**2 / L**2)* cos(2.0*math.pi*x[0]/L) )

u = TrialFunction(Vcg)
phi2 = TestFunction(Vcg)


Ab = (inner(grad(u), grad(phi2))) * dx
Lb = -g * phi2 * dx

b_problem = LinearVariationalProblem(Ab, Lb, b0)
b_solver = LinearVariationalSolver(b_problem, nullspace=nullspace, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_sub_type': 'ilu'
    # 'ksp_type':'cg',
    # 'pc_type':'sor'
})

# solve and output
w_solver.solve()
b_solver.solve()

diff = Function(Vcg)
diff.assign(b0 - w0)

# try:
#     import matplotlib.pyplot as plt
# except:
#     warning("Matplotlib not imported")

# try:
#     fig, axes = plt.subplots()
#     colors = tripcolor(diff, axes=axes)
#     fig.colorbar(colors)
# except Exception as e:
#     warning("Cannot plot figure. Error msg '%s'" % e)

# try:
#     plt.savefig('initial_buoyance.png')
#     # plt.show()
# except Exception as e:
#     warning("Cannot show figure. Error msg '%s'" % e)


exact = Function(Vcg)
exact.rename("exact_particular_sol")
exact.interpolate(-(L**2/2.0/math.pi**2 + 2.0)* (cos(2.0*math.pi*x[0]/L) - 1))

b0.rename("from_specified_g")
w0.rename("from_grad_g")
diff.rename("diff")
file = File('initial_buoyancy.pvd')
file.write(diff, b0, w0, exact)
