# halo vec_wo for f_reduce, vec_ro for f
# non-blocking isend and irecv
# wait_all
# don't use camel case

from firedrake import * 

import sys
n0 =int(sys.argv[1]) 
T = float(sys.argv[2])

my_ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
ensemble_comm = my_ensemble.ensemble_comm   # M parameter is spatial
comm = my_ensemble.comm

nprocs = ensemble_comm.size
procno = ensemble_comm.rank

cg_deg = 1
Lx   = 1                                     # Zonal length
Ly   = 1                                     # Meridonal length
mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="both", comm=comm) 
#if (procno == 0):
#    print("proc: {}, total {}, experiment: ".format(procno,nprocs), n0, T     )

# We define function spaces::
Vdg = FunctionSpace(mesh,"DG",1)
Vcg = FunctionSpace(mesh,"CG",cg_deg)
Vu  = VectorFunctionSpace(mesh,"DG",1)

x = SpatialCoordinate(mesh)
# stationary Euler on a torus
q0 = Function(Vdg).interpolate(sin(8.0*pi*x[0])*sin(8.0*pi*x[1]) +
              0.4*cos(6.0*pi*x[0])*cos(6.0*pi*x[1])+
              0.3*cos(10.0*pi*x[0])*cos(4.0*pi*x[1]) +
              0.02*sin(2.0*pi*x[1]) + 0.02*sin(2.0*pi*x[0]))

initial = Function(Vdg).assign(q0)

dq1 = Function(Vdg)       # PV fields for different time steps
qh  = Function(Vdg)
q1  = Function(Vdg)

psi0 = Function(Vcg)      # Streamfunctions for different time steps

dts = {50: 0.01, 64: 0.01, 128: 0.005, 256:0.005, 512: 0.0025}
F    = Constant(0.0)         # Rotational Froude number
beta = Constant(0.0)         # beta plane coefficient
Dt = dts[n0]
dt   = Constant(Dt)
dumpf = {50: 10, 64:10, 128:20, 512: 40, 256:20}

psi = TrialFunction(Vcg)
phi = TestFunction(Vcg)


Apsi = (inner(grad(psi),grad(phi)))*dx
Lpsi = -q1*phi*dx

bc1 = []

psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0,bcs=bc1)
psi_solver = LinearVariationalSolver(psi_problem,
                                     solver_parameters={
        'ksp_type':'cg',
        'pc_type':'sor'
        })

gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

n = FacetNormal(mesh)
un = 0.5*(dot(gradperp(psi0), n) + abs(dot(gradperp(psi0), n)))

_un_ = dot(gradperp(psi0), n)
_abs_un_ = abs(_un_)

q = TrialFunction(Vdg)
p = TestFunction(Vdg)
a_mass = p*q*dx
a_int  = (dot(grad(p), -gradperp(psi0)*q))*dx

a_flux = 0.5*jump(p)*(2*_un_('+')*avg(q) + _abs_un_('+')*jump(q))*dS  
arhs = a_mass - dt*(a_int+a_flux)
q_problem = LinearVariationalProblem(a_mass, action(arhs,q1), dq1)
q_solver  = LinearVariationalSolver(q_problem,
                                    solver_parameters={
        'ksp_type':'preonly',
        'pc_type':'bjacobi',
        'sub_pc_type': 'ilu'
        })
                                    
output_file = None
if (procno == 0):
    output_file = File("/home/wpan1/Data/PythonProjects/tqg_parallel/output.pvd")
    output_file.write(q0, time=0)

t =0.
dump = 0
while(t < (T-Dt/2)):

  # Compute the streamfunction for the known value of q0
  q1.assign(q0)
  psi_solver.solve()
  q_solver.solve()

  # Find intermediate solution q^(1)
  q1.assign(dq1)
  psi_solver.solve()
  q_solver.solve()

  # Find intermediate solution q^(2)
  q1.assign(0.75*q0 + 0.25*dq1)
  psi_solver.solve()
  q_solver.solve()

  # Find new solution q^(n+1)
  q0.assign(q0/3 + 2*dq1/3)

  t += Dt
  dump += 1

  if dump == dumpf[n0]:
      _t = round(t,4)
      dump -= dumpf[n0]
      if (procno == 0):
          output_file.write(q0, time=_t)
      if (comm.rank == 0):
          print(_t,flush=True)

