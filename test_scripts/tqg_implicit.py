from firedrake import *
from utility import Workspace

workspace = "/home/wpan1/Data/PythonProjects/tqg_example3_implicit/"

import sys
n0 = int(sys.argv[1])
T = int(sys.argv[2])
Lx   = 1 
Ly   = 1 
mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="both") 

print("experiment: ", n0)

# constants ##########################################
Dt = 0.001
dt   = Constant(Dt)
dumpfreq = 100

# FEM spaces #########################################
Vdg = FunctionSpace(mesh,"DG",1)
Vcg = FunctionSpace(mesh,"CG",1)
Vmixed = MixedFunctionSpace((Vdg,Vdg, Vcg)) # omega, b, psi
bathymetry = Function(Vcg).assign(0)
rotation = Function(Vdg).assign(0)
x = SpatialCoordinate(mesh)

w0 = Function(Vmixed)
w1 = Function(Vmixed)

q0,b0, psi0 = w0.split()

# stationary initial condition for Euler on flat torus
q0.interpolate(  cos(2.0*pi*x[1]) ) 
b0.interpolate( sin(2.0*pi*x[1])-1 )
initial = Function(Vdg).assign(q0)
initial_b = Function(Vdg).assign(b0)

# get initial psi ####################################

psi = TrialFunction(Vcg)
phi = TestFunction(Vcg)

Apsi = (inner(grad(psi),grad(phi))+psi*phi)*dx
Lpsi = (rotation-q0)*phi*dx

bc1 = []

psi_solver = LinearVariationalSolver(LinearVariationalProblem(Apsi,Lpsi,psi0,bcs=bc1),
                                     solver_parameters={
        'ksp_type':'cg',
        'pc_type':'sor'
        })
psi_solver.solve()

# weak form of the the time stepping #################
#p for omega, v for b, phi for psi
p,v, phi = TestFunctions(Vmixed)

w1.assign(w0)  # initial guess?
q1, b1,psi1 = split(w1)
q0, b0,psi0 = split(w0)

qh = 0.5*(q1 + q0)
psih = 0.5*(psi1 + psi0)
bh = 0.5*(b1 + b0)

gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

n = FacetNormal(mesh)
un = 0.5*(dot(gradperp(psih), n) + abs(dot(gradperp(psih), n)))
_un_ = dot(gradperp(psih), n)
_abs_un_ = abs(_un_)

L = ( (inner(grad(psi1),grad(phi)) + psi1*phi + q1*phi - rotation*phi)*dx +
      (p*(q1 - q0) - dt * dot(grad(p), gradperp(psih) * (qh-bh))
      + 0.5*dt*div( bh* gradperp( bathymetry )) * p 
          )*dx
      + dt * (0.5*jump(p)*( 2*_un_('+')*avg(qh-bh) + _abs_un_('+')*jump(qh-bh)   ) ) *dS
        # + dt * dot(jump(p),  un('+')*(qh-bh)('+') - un('-')*(qh-bh)('-')) *dS # upwinding
     +( v*(b1 - b0) - dt* dot(grad(v), gradperp(psih) * bh) )*dx
     +  dt * (0.5*jump(v)*( 2*_un_('+')*avg(bh) + _abs_un_('+')*jump(bh) ) ) *dS
     # + dt * dot(jump(v), un('+')*bh('+') - un('-')*bh('-'))*dS
)


qprob = NonlinearVariationalProblem(L, w1)
qsolver = NonlinearVariationalSolver(qprob,
                solver_parameters={"snes_type": "newtonls"}) #,
#                                'snes_monitor': None,
#                   'snes_view': None,
#                   'ksp_monitor_true_residual': None,
#                   'snes_converged_reason': None,
#                   'ksp_converged_reason': None
#                   })
                                 # "ksp_type" : "preonly",
                                 # "pc_type" : "lu"}) #, 
# solver_parameters={                                 
    # 'mat_type': 'aij',
    # 'ksp_type': 'preonly',
    # 'pc_type': 'lu'
# })

# run and output #######################################################
q0,b0, psi0 = w0.split()
q1,b1, psi1 = w1.split()

output = File(workspace + "visuals/output_implicitMid_{}.pvd".format(n0))

t = 0.
tdump = 0

output.write(q1,b1, psi1, time=round(t, 2))

q_errors =[]
b_errors = []
while (t < (T-Dt*0.5)):

  t += Dt
  qsolver.solve()
  w0.assign(w1)
  
  tdump += 1
  if tdump == dumpfreq:
      tdump -= dumpfreq
      t_ = round(t, 2)
      print(t_)
      output.write(q1,b1, psi1, time=t_)
      q_errors.append(errornorm(q1, initial, norm_type='L1'))
      b_errors.append(errornorm(b1, initial_b, norm_type='L1'))

workspace = Workspace(workspace + "plots")
np.save(workspace.output_name("q_errors"), np.asarray(q_errors))
np.save(workspace.output_name("b_errors"), np.asarray(b_errors))

try:
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots()
  axes.plot(q_errors, c='r', label='q errors')
  axes.plot(b_errors, c='b', label='b errors')

  plt.legend()
  plt.savefig(workspace.output_name("tqg_error_implicitMid_{}.png".format(n0)),dpi=100)
  plt.close()
except:
  warning("plot failed")

