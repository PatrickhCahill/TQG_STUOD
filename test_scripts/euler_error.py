from firedrake import *
from utility import Workspace

workspace = "/home/wpan1/Data/PythonProjects/"

import sys
fluxchoice = str(sys.argv[1]) # 'central'
n0 = int(sys.argv[2])
# cg_deg = int(sys.argv[3])
cg_deg = 1

Lx   = 1                                     # Zonal length
Ly   = 1                                     # Meridonal length
# n0   = 50                                    # Spatial resolution
mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="both") 
# quadrilateral=True)

print("experiment: ", fluxchoice, n0, cg_deg)

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

#(sin(2*pi*x[0]) * sin(2*pi*x[1]))
initial = Function(Vdg).assign(q0)

dq1 = Function(Vdg)       # PV fields for different time steps
qh  = Function(Vdg)
q1  = Function(Vdg)

psi0 = Function(Vcg)      # Streamfunctions for different time steps

dts = {50: 0.01, 64: 0.005, 128: 0.001}
dumpfs = {50: 50, 64: 200, 128: 1000}

F    = Constant(0.0)         # Rotational Froude number
beta = Constant(0.0)         # beta plane coefficient
# Dt   = dts[n0]                  # Time step
# dumpfreq = dumpfs[n0]
Dt = dts[50]
dumpfreq = dumpfs[50]
dt   = Constant(Dt)

psi = TrialFunction(Vcg)
phi = TestFunction(Vcg)


Apsi = (inner(grad(psi),grad(phi)))*dx
Lpsi = -q1*phi*dx


# bc1 = DirichletBC(Vcg, 0., 'on_boundary')
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

if (fluxchoice == 'central'):
  a_flux = (p('+') * _un_('+') + p('-') * _un_('-') ) * avg(q) * dS  # central flux

elif fluxchoice == 'upwinding':
  a_flux = (dot(jump(p), un('+')*q('+') - un('-')*q('-')) )*dS  # upwinding

elif fluxchoice == 'lf':
    # LF form of upwinding -- the actual LF is exactly upwinding
    a_flux = 0.5*jump(p)*(2*_un_('+')*avg(q) + _abs_un_('+')*jump(q))*dS  

    # LF, this is also equal to the above, 
    # if _un_ is continuous, the conditional statement here is unnecessary
    # a_flux = ( (p('+') * _un_('+') + p('-') * _un_('-') ) * avg(q) + \
            # 0.5 * jump(p) * conditional( _abs_un_('+') > _abs_un_('-'),\
            # _abs_un_('+'), _abs_un_('-') ) * jump(q) ) * dS

arhs   = a_mass - dt*(a_int + a_flux)

q_problem = LinearVariationalProblem(a_mass, action(arhs,q1), dq1)

q_solver  = LinearVariationalSolver(q_problem,
                                    solver_parameters={
        'ksp_type':'preonly',
        'pc_type':'bjacobi',
        'sub_pc_type': 'ilu'
        })


q0.rename("Potential vorticity")
psi0.rename("Stream function")
v = Function(Vu, name="gradperp(stream function)")
v.project(gradperp(psi0))

output = File(workspace + "euler_error_n{}_cg{}/visuals/output{}.pvd".format(n0, cg_deg, fluxchoice))

output.write(q0, psi0, v)

t = 0.
T = 50.
tdump = 0

v0 = Function(Vu)

q_errors = []
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

  tdump += 1
  norm_choice = 'L1'
  if tdump == dumpfreq:
      tdump -= dumpfreq
      v.project(gradperp(psi0))
      t_ = round(t, 2)
      print(t_)
      output.write(q0, psi0, v, time=t_)
      q_errors.append(errornorm(q0, initial, norm_type=norm_choice))

# try:
  # import matplotlib.pyplot as plt
  # plt.plot(q_errors)
  # workspace = Workspace(workspace + "euler_error_n{}_cg{}/plots".format(n0, cg_deg))
  # plt.savefig(workspace.output_name("euler_error_{}_{}.png".format(fluxchoice, norm_choice)), dpi=300)
  # plt.close()
# except:
  # warning("plot failed")
