from firedrake import *

Lx   = 1                                     # Zonal length
Ly   = 1                                     # Meridonal length
n0   = 64                                       # Spatial resolution
mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="both") # quadrilateral=True)

# We define function spaces::

Vdg = FunctionSpace(mesh,"DG",1)               # DG elements for Potential Vorticity (PV)
Vcg = FunctionSpace(mesh,"CG",1)               # CG elements for Streamfunction


x = SpatialCoordinate(mesh)
q1 = Function(Vdg).interpolate(sin(2*pi*x[0]) * sin(2*pi*x[1]))  # stationary Euler on a torus

psi0 = Function(Vcg)      # Streamfunctions for different time steps
psi = TrialFunction(Vcg)
phi = TestFunction(Vcg)

# Build the weak form for the inversion
Apsi = (inner(grad(psi),grad(phi)))*dx
Lpsi = -q1*phi*dx

# We impose homogeneous dirichlet boundary conditions on the stream
# function at the top and bottom of the domain. ::

# bc1 = DirichletBC(Vcg, 0., 'on_boundary')
bc1 = []

psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0,bcs=bc1)
psi_solver = LinearVariationalSolver(psi_problem,
                                     solver_parameters={
        'ksp_type':'cg',
        'pc_type':'sor'
        })


psi_solver.solve()

File("/home/wpan1/Documents/Data2TB/WPan1/poisson/poisson.pvd").write(psi0, q1)

try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

try:
  fig, axes = plt.subplots()
  colors = tripcolor(psi0, axes=axes)
  fig.colorbar(colors)
except Exception as e:
  warning("Cannot plot figure. Error msg '%s'" % e)

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg '%s'" % e)  
