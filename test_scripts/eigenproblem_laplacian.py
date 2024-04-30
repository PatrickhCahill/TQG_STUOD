from firedrake import *

Lx   = 1                                     # Zonal length
Ly   = 1                                     # Meridonal length
n0   = 128                                       # Spatial resolution
mesh = UnitSquareMesh(n0,n0)
Vcg  = FunctionSpace(mesh, "CG", 2)               # CG elements for Streamfunction
x = SpatialCoordinate(mesh)

psi0 = Function(Vcg)      # Streamfunctions for different time steps

F    = Constant(2*pi*pi)         # Rotational Froude number
beta = Constant(0.0)

psi = TrialFunction(Vcg)
phi = TestFunction(Vcg)


bc = DirichletBC(Vcg, 0, 'on_boundary')

Apsi = (inner(grad(psi),grad(phi)))*dx 
Lpsi = -F*psi*phi*dx  #beta * phi*dx

A = assemble(Apsi, bcs=bc).M.handle
M = assemble(Lpsi).M.handle

opts = PETSc.Options()
opts.setValue('eps_gen_hermitian', None)
opts.setValue('eps_target_real', None)
opts.setValue('eps_smallest_real', None)
opts.setValue('st_type', 'sinvert')
opts.setValue('st_ksp_type', 'cg')
opts.setValue('st_pc-type', 'jacobi')
opts.setValue('eps_tol', 1e-8)

from  slepc4py import SLEPc
num_values = 250
eigensolver = SLEPc.EPS().create(comm=firedrake.COMM_WORLD)
eigensolver.setDimensions(num_values)
eigensolver.setOperators(A, M)
eigensolver.setFromOptions()
eigensolver.solve()

num_converged = eigensolver.getConverged()
print(num_converged)

# # psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0, bcs=DirichletBC(Vcg, 0., 'on_boundary'))
# # psi_solver = LinearVariationalSolver(psi_problem, solver_parameters={'ksp_type':'cg', 'pc_type':'sor'})
# solve(Apsi == Lpsi, psi0, solver_parameters={'ksp_type':'cg', 'pc_type':'sor'})

# # output = File("/home/wpan1/Documents/Data2TB/WPan1/euler_error{}/output{}.pvd".format(n0, fluxchoice))

# # psi_solver.solve()

# exact_solution = Function(Vcg).interpolate( sin(pi*x[0])*sin(pi*x[1]) )

# # output.write(psi0)

# print(norm(psi0))
# print( assemble(abs(exact_solution - psi0)*dx) )
# print( assemble(abs(exact_solution)*dx) )
