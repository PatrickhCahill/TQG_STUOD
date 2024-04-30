#
#   Author: Wei Pan
#   Copyright   2017
#
#   euler.py
#
#   Defines OOP classes for the Euler model and corresponding solver
#

from firedrake import *
import numpy as np
from tqg.diagnostics import tqg_mesh_grid


class EulerFemParams(object):
    """ firedrake finite element parameters """

    def __init__(self, dt, msh):

        self.Vdg = FunctionSpace(msh, "DG", 1)
        self.Vcg = FunctionSpace(msh, "CG", 1)
        self.Vu = VectorFunctionSpace(msh, "DG", 1)
        self.x = SpatialCoordinate(msh)
        self.dt = dt
        self.facet_normal = FacetNormal(msh)


class EulerModelParams(object):
    """ 2D Euler model parameters """

    def __init__(self, fem_params, t):

        # ## slip boundary condition
        self.bcs = [] # DirichletBC(fem_params.Vcg, 0.0, 'on_boundary')
        self.time_length = t
        self.r = Constant(0.0)


class EulerParams(object):

    """ Wrapper class for EulerModelParams and EulerFemParams """
    def __init__(self, t, dt, msh):
        self.euler_fem_params = EulerFemParams(dt, msh)
        self.euler_model_params = EulerModelParams(self.euler_fem_params, t)
        self.initial_cond = Function(self.euler_fem_params.Vdg)

        x= SpatialCoordinate(self.euler_fem_params.Vdg.mesh())
        self.initial_cond.interpolate(sin(8.0*pi*x[0])*sin(8.0*pi*x[1]) +
                                      0.4*cos(6.0*pi*x[0])*cos(6.0*pi*x[1])+
                                      0.3*cos(10.0*pi*x[0])*cos(4.0*pi*x[1]) +
                                      0.02*sin(2.0*pi*x[1]) + 0.02*sin(2.0*pi*x[0]))

   
class EulerSolver(object):

    def __init__(self, euler_params):

        self.gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))

        fem_params = euler_params.euler_fem_params
        model_params = euler_params.euler_model_params

        self.time_length = model_params.time_length
        self.initial_cond = euler_params.initial_cond

        self.Vdg = fem_params.Vdg
        Vdg = fem_params.Vdg
        Vcg = fem_params.Vcg
        self.Vu = fem_params.Vu

        self.dq1 = Function(Vdg)  # PV fields for different time steps        # next time step
        self.q1 = Function(Vdg)
        self.psi0 = Function(Vcg)  # Streamfunctions for different time steps

        damp_rate = model_params.r
        self.Dt = fem_params.dt
        dt = Constant(self.Dt)

        forcing = Function(Vdg)
        x= SpatialCoordinate(Vdg.mesh())
        # forcing.interpolate(0.1*sin(8.0*math.pi*x[0]))
        forcing.assign(0)

        psi = TrialFunction(Vcg)
        phi = TestFunction(Vcg)

        # --- Elliptic equation ---
        # Build the weak form for the inversion
        # stream function is 0 on the boundary - top and bottom, left and right
        Apsi = (inner(grad(psi), grad(phi))) * dx
        Lpsi = -self.q1 * phi * dx

        self.psi_problem = LinearVariationalProblem(Apsi, Lpsi, self.psi0, model_params.bcs)
        self.psi_solver = LinearVariationalSolver(self.psi_problem, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_sub_type': 'ilu'
            # 'ksp_type':'cg',
            # 'pc_type':'sor'
        })

        un = 0.5 * (dot(self.gradperp(self.psi0), fem_params.facet_normal) +
                    abs(dot(self.gradperp(self.psi0), fem_params.facet_normal)))

        q = TrialFunction(Vdg)
        p = TestFunction(Vdg)
        a_mass = p * q * dx
        a_int = (dot(grad(p), -self.gradperp(self.psi0) * q) - p * (forcing - q * damp_rate)) * dx

        a_flux = (dot(jump(p), un('+') * q('+') - un('-') * q('-'))) * dS
        arhs = a_mass - dt * (a_int + a_flux)

        self.q_problem = LinearVariationalProblem(a_mass, action(arhs, self.q1), self.dq1)  # solve for dq1

        self.q_solver = LinearVariationalSolver(self.q_problem, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        })

        
    def solve(self, dumpfreq, output_name, res):
        q0 = Function(self.Vdg)
        q0.assign(self.initial_cond)

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")
        v = Function(Vu, name="Velocity")
        v.project(self.gradperp(self.psi0))

        output_file = File(output_name + ".pvd")
        output_file.write(q0, self.psi0, v)
       
        t = 0.
        T = self.time_length
        tdump = 0

        mesh_grid = tqg_mesh_grid(res)
        index = 0

        while t < (T - Dt / 2):

            # Compute the streamfunction for the known value of q0
            self.q1.assign(q0)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(1)
            self.q1.assign(self.dq1)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(2)
            self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find new solution q^(n+1)
            q0.assign(q0 / 3 + 2 * self.dq1 / 3)

            # Store solutions to xml and pvd
            t += Dt
            tdump += 1
            if tdump == dumpfreq:
                tdump -= dumpfreq
                _t = round(t, 2)
                v.project(self.gradperp(self.psi0))
                output_file.write(q0, self.psi0, v, time=_t)
                
                # ---- FFT ------
                index += 1
                np.save(output_name + "_{}".format(index), np.asarray(v.at(mesh_grid, tolerance=1e-10)))
                # ---- end fft ----
                print(_t)

        # print("----%s seconds------" % (time.time() - start_time))
        self.initial_cond.assign(q0)
