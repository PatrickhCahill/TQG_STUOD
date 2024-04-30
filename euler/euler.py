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
import utility
import math


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

        # set bcs = [] if you don't want any boundary conditions
        #bcv = cos(2. *pi) + 0.5 * cos(4.0*pi ) + cos(6.0*pi)/3.0
        bcv = 1.5 + 1./3.
        self.bcs = DirichletBC(fem_params.Vcg, bcv, 'on_boundary')
        self.time_length = t
        self.r = Constant(0.1) 


class EulerParams(object):

    """ Wrapper class for EulerModelParams and EulerFemParams """
    def __init__(self, t, dt, msh):
        self.euler_fem_params = EulerFemParams(dt, msh)
        self.euler_model_params = EulerModelParams(self.euler_fem_params, t)
        self.initial_cond = Function(self.euler_fem_params.Vdg)

    def set_initial_condition_hardcoded(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        self.initial_cond.interpolate(Expression("sin(2*pi*(x[0]-0.25))"))

    def set_initial_condition_from_chkpoint(self, filename):
        """ set the initial condition from chkpoint file"""
        DumbCheckpoint(filename, mode=FILE_READ).load(self.initial_cond, name="Vorticity")

    def get_q(self):
        return self.initial_cond


class EulerSolver(object):

    def __init__(self, euler_params):

        self.base_time = 0

        fem_params = euler_params.euler_fem_params
        model_params = euler_params.euler_model_params

        self.time_length = model_params.time_length
        self.initial_cond = Function(fem_params.Vdg)

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
        #forcing.interpolate(sin(8.0*math.pi*(x[0]+x[1])))
        forcing.interpolate( 10*cos(3.*math.pi*x[1])*(cos(2. * math.pi * (x[0]) ) + 0.5 * cos(4.*math.pi*x[0]) + cos(6.*math.pi*(x[0]))/3. ))
        #forcing.assign(0)

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

        # Next we'll set up the advection equation, for which we need an
        # operator :math:`\vec\nabla^\perp`, defined as a python anonymouus
        # function::

        # gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

        # For upwinding, we'll need a representation of the normal to a facet,
        # and a way of selecting the upwind side::

        un = 0.5 * (dot(self.gradperp(self.psi0), fem_params.facet_normal) +
                    abs(dot(self.gradperp(self.psi0), fem_params.facet_normal)))

        # Now the variational problem for the advection equation itself. ::

        q = TrialFunction(Vdg)
        p = TestFunction(Vdg)

        a_mass = p * q * dx

        a_int = (dot(grad(p), -self.gradperp(self.psi0) * q) - p * (forcing - q * damp_rate)) * dx

        # DG related term, this does not exist in the CG version
        a_flux = (dot(jump(p), un('+') * q('+') - un('-') * q('-'))) * dS
        arhs = a_mass - dt * (a_int + a_flux)

        # here the a_mass corresponds to <q^1,p>
        # a_mass(u,v) = L(v), solve for u, and store in dq1 i think.
        self.q_problem = LinearVariationalProblem(a_mass, action(arhs, self.q1), self.dq1)  # solve for dq1

        # Since the operator is a mass matrix in a discontinuous space, it can
        # be inverted exactly using an incomplete LU factorisation with zero
        # fill. ::
        self.q_solver = LinearVariationalSolver(self.q_problem, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        })

    @staticmethod
    def gradperp(u):
        return as_vector((-u.dx(1), u.dx(0)))

    @staticmethod
    def helmholtz_solver(noise, k_sqr=1.0):
        """ 
        solves a helmholtz problem given a function with i.i.d.
        randomly drawn basis coefficients

        assumes alpha = 1   (I - c. \Laplace)noise_bar = noise
        solving for noise_bar which is like an averaged version of noise
        
        this averages out frequencies which are less than c.
        """
        fs = noise.function_space()
        cg_fs = FunctionSpace(fs.mesh(), 'CG', 1)
        f = Function(cg_fs).project(noise)

        u = TrialFunction(cg_fs)
        v = TestFunction(cg_fs)
        c_sqr = Constant(1.0 / k_sqr)

        a = (c_sqr * dot(grad(v), grad(u)) + v * u) * dx
        L = f * v * dx

        bc = DirichletBC(cg_fs, 0.0, 'on_boundary')

        u = Function(cg_fs)
        solve(a == L, u, bcs=bc, solver_parameters={'ksp_type': 'cg'})

        return noise.project(u)  ## note if noise here is dg then u get projected to dg

    @staticmethod
    def solve_for_q_given_psi(psi, q_dg):
        """ given streamfunction, solve for the corresponding vorticity

            laplace(psi) = q
        """
        msh = psi.function_space().mesh()
        q_cg_space = FunctionSpace(msh, 'CG', 1)

        q = TrialFunction(q_cg_space)
        v = TestFunction(q_cg_space)

        #bc = DirichletBC(q_cg_space, 0.0, 'on_boundary')

        a = -(dot(grad(v), grad(psi))) * dx
        L = q * v * dx

        q0 = Function(q_cg_space)
        solve(L == a, q0) #, bcs=[bc])
        q_dg.project(q0)

    def psi_solve_given_q(self, pv):
        """ solve for streamfunction given vorticity

            laplace(psi) = q
        """
        self.q1.assign(pv)
        self.psi_solver.solve()

    def v_given_q(self, pv):
        """
        compute velocity vector given vorticity
        :param pv:
        :return:
        """
        self.q1.assign(pv)
        self.psi_solver.solve()
        return self.gradperp(self.psi0)

    def solver(self, dumpfreq, _q0, output_name, output_visual_flag=False, chkpt_flag=False):
        """
        solver for the whole euler system given initial condition q0

        :param dumpfreq:
        :param _q0:
        :param output_name: name of output files, stored in utility.output_directory
        :param output_visual_flag: if True, this function will output pvd files for visualisation at a frequency
        defined by dumpfreq
        :param chkpt_flag: if True, this function will store solved q as chkpoint file at solution times defined by
         dumpfreq
        :return: 
        """
        # assume q0 is given in initial_cond
        q0 = Function(self.Vdg)
        q0.assign(_q0)

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")
        v = Function(Vu, name="Velocity")
        v.project(self.gradperp(self.psi0))

        output_file = File(output_name + ".pvd")

        with DumbCheckpoint(output_name, single_file=False, mode=FILE_CREATE) as chk_q:

            if output_visual_flag:
                output_file.write(q0, self.psi0, v, time=0)

            if chkpt_flag:
                chk_q.store(q0)  # store initial as chkpoint

            # Now all that is left is to define the timestepping parameters and
            # execute the time loop. ::

            t = 0.
            T = self.time_length
            tdump = 0

            # start_time = time.time()
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
                    _t = round(t, 5)

                    if output_visual_flag:
                        v.project(self.gradperp(self.psi0))
                        output_file.write(q0, self.psi0, v, time=self.base_time+_t)

                    if chkpt_flag:
                        chk_q.new_file()
                        chk_q.store(q0)

                    print(_t, flush=True)

            # print("----%s seconds------" % (time.time() - start_time))
            self.initial_cond.assign(q0)

    def pdesolver(self, _q0, output_name, chkpt_flag=False):
        """
        solver for the whole euler system given initial condition q0

        removes dumpfreq, used by particlefilter

        :param dumpfreq:
        :param _q0:
        :param output_name: name and directory of output chkpt file
        :param chkpt_flag:
        :return:
        """
        # assume q0 is given in initial_cond
        q0 = Function(self.Vdg)
        q0.assign(_q0)

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")
        v = Function(Vu, name="Velocity")
        v.project(self.gradperp(self.psi0))

        # Now all that is left is to define the timestepping parameters and
        # execute the time loop. ::

        t = 0.
        T = self.time_length

        # start_time = time.time()
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

        # print("----%s seconds------" % (time.time() - start_time))
        self.initial_cond.assign(q0)

        if chkpt_flag:
            with DumbCheckpoint(output_name, mode=FILE_CREATE, single_file=True) as chk:
                chk.store(q0)
