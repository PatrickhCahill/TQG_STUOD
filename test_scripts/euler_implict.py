from firedrake import *
from utility import Workspace

workspace = "/home/wpan1/Data/PythonProjects/"

import sys
n0 = int(sys.argv[1])

Lx   = 1 
Ly   = 1 
mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="both") 

print("experiment: ", n0)

# constants ##########################################
Dt = 0.1
dt   = Constant(Dt)
dumpfreq = 1
T = 50

# FEM spaces #########################################
Vdg = FunctionSpace(mesh,"DG",1)
Vcg = FunctionSpace(mesh,"CG",1)
Vmixed = MixedFunctionSpace((Vdg, Vcg))

x = SpatialCoordinate(mesh)

w0 = Function(Vmixed)
w1 = Function(Vmixed)

q0, psi0 = w0.split()

# stationary initial condition for Euler on flat torus
# q0.interpolate(sin(2*pi*x[0]) * sin(2*pi*x[1])) 
q0.interpolate(sin(8.0*pi*x[0])*sin(8.0*pi*x[1]) +
              0.4*cos(6.0*pi*x[0])*cos(6.0*pi*x[1])+
              0.3*cos(10.0*pi*x[0])*cos(4.0*pi*x[1]) +
              0.02*sin(2.0*pi*x[1]) + 0.02*sin(2.0*pi*x[0]))

initial = Function(Vdg).assign(q0)

# get initial psi ####################################

psi = TrialFunction(Vcg)
phi = TestFunction(Vcg)

Apsi = (inner(grad(psi),grad(phi)))*dx
Lpsi = -q0*phi*dx

bc1 = []

psi_solver = LinearVariationalSolver(LinearVariationalProblem(Apsi,Lpsi,psi0,bcs=bc1),
                                     solver_parameters={
        'ksp_type':'cg',
        'pc_type':'sor'
        })
psi_solver.solve()

# weak form of the the time stepping #################
p, phi = TestFunctions(Vmixed)

w1.assign(w0)  # initial guess?
q1, psi1 = split(w1)
q0, psi0 = split(w0)

qh = 0.5*(q1 + q0)
psih = 0.5*(psi1 + psi0)

gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

n = FacetNormal(mesh)
un = 0.5*(dot(gradperp(psih), n) + abs(dot(gradperp(psih), n)))
_un_ = dot(gradperp(psih), n)

L = ( (inner(grad(psi1),grad(phi)) + q1*phi)*dx +
      (p*(q1 - q0) - dt * dot(grad(p), gradperp(psih) * qh))*dx
        + dt * dot(jump(p),  un('+')*qh('+') - un('-')*qh('-')) *dS # upwinding
)


qprob = NonlinearVariationalProblem(L, w1)
qsolver = NonlinearVariationalSolver(qprob,
                solver_parameters={"snes_type": "newtonls",
                                'snes_monitor': None,
                   'snes_view': None,
                   'ksp_monitor_true_residual': None,
                   'snes_converged_reason': None,
                   'ksp_converged_reason': None
                   })
                                 # "ksp_type" : "preonly",
                                 # "pc_type" : "lu"}) #, 
# solver_parameters={                                 
    # 'mat_type': 'aij',
    # 'ksp_type': 'preonly',
    # 'pc_type': 'lu'
# })

# run and output #######################################################
q0, psi0 = w0.split()
q1, psi1 = w1.split()

output = File(workspace + "euler_implicit/visuals/output_implicitMid_{}.pvd".format(n0))

t = 0.
tdump = 0

output.write(q1, psi1, time=round(t, 2))

q_errors =[]
while (t < (T-Dt*0.5)):

  t += Dt
  qsolver.solve()
  w0.assign(w1)
  
  tdump += 1
  if tdump == dumpfreq:
      tdump -= dumpfreq
      t_ = round(t, 2)
      print(t_)
      output.write(q1, psi1, time=t_)
      q_errors.append(errornorm(q1, initial, norm_type='L1'))

try:
  import matplotlib.pyplot as plt
  plt.plot(q_errors)
  workspace = Workspace(workspace + "euler_implicit/plots")
  plt.savefig(workspace.output_name("euler_error_implicitMid_{}.png".format(n0)), dpi=100)
  plt.close()
except:
  warning("plot failed")
