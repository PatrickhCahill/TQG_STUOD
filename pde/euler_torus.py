from firedrake import *
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
    """ TQG model parameters """

    def __init__(self, fem_params, t):
        # ## slip boundary condition
        self.time_length = t
        self.r = Constant(0)


class EulerParams(object):
    """ Wrapper class for EulerModelParams and EulerFemParams """

    def __init__(self, T, dt, msh):
        self.euler_fem_params = EulerFemParams(dt, msh)
        self.euler_model_params = EulerModelParams(self.euler_fem_params, T)
        self.initial_cond = Function(self.euler_fem_params.Vdg)

    def set_initial_condition_hardcoded(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.euler_fem_params.Vdg.mesh())
        self.initial_cond.interpolate(sin(2.0*math.pi*x[0])+cos(2.0*math.pi*x[1]))
        # self.initial_cond.interpolate(sin(8.0*pi*x[0])*sin(8.0*pi*x[1]) +
        #                               0.4*cos(6.0*pi*x[0])*cos(6.0*pi*x[1])+
        #                               0.3*cos(10.0*pi*x[0])*cos(4.0*pi*x[1]) +
        #                               0.02*sin(2.0*pi*x[1]) + 0.02*sin(2.0*pi*x[0]))

        return self.initial_cond


class EulerSolver(object):

    def __init__(self, euler_params):
        self.base_time = 0

        fem_params = euler_params.euler_fem_params
        model_params = euler_params.euler_model_params

        self.time_length = model_params.time_length
        self.initial_cond = euler_params.set_initial_condition_hardcoded()

        self.Vdg = fem_params.Vdg
        Vdg = fem_params.Vdg
        Vcg = fem_params.Vcg
        self.Vu = fem_params.Vu

        self.dq1 = Function(Vdg)  # PV fields for different time steps   # next time step
        self.q1 = Function(Vdg)
        self.psi0 = Function(Vcg)  # Streamfunctions for different time steps

        damp_rate = model_params.r
        self.Dt = fem_params.dt
        dt = Constant(self.Dt)

        forcing = Function(Vdg)
        x= SpatialCoordinate(Vdg.mesh())
        # forcing.interpolate(0.1*sin(8.0*math.pi*x[0]))
        forcing.interpolate(0*x[0])

        psi = TrialFunction(Vcg)
        phi = TestFunction(Vcg)

        # --- Elliptic equation ---
        # Build the weak form for the inversion
        # stream function is 0 on the boundary - top and bottom, left and right
        Apsi = (inner(grad(psi), grad(phi))) * dx
        Lpsi = -self.q1 * phi * dx

        nullspace = VectorSpaceBasis(constant=True)
        self.psi_problem = LinearVariationalProblem(Apsi, Lpsi, self.psi0)
        self.psi_solver = LinearVariationalSolver(self.psi_problem, nullspace=nullspace)

        # Next we'll set up the advection equation, for which we need an
        # operator :math:`\vec\nabla^\perp`, defined as a python anonymouus
        # function::

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
        self.q_solver = LinearVariationalSolver(self.q_problem)

    @staticmethod
    def gradperp(u):
        return as_vector((-u.dx(1), u.dx(0)))
    
    def solve(self, dumpfreq, output_name):
        """
        solve for the whole euler system given initial condition q0

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
        q0.assign(self.initial_cond)

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")
        v = Function(Vu, name="Velocity")
        v.project(self.gradperp(self.psi0))

        output_file = File(output_name + ".pvd")
        output_file.write(q0, self.psi0, v)

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
                _t = round(t, 2)

                v.project(self.gradperp(self.psi0))
                output_file.write(q0, self.psi0, v, time=self.base_time+_t)

                print(_t)

        # print("----%s seconds------" % (time.time() - start_time))
        self.initial_cond.assign(q0)
