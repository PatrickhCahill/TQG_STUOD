from firedrake import *
import numpy as np
from linearised_tqg.diagnostics import tqg_energy, tqg_kinetic_energy, tqg_potential_energy


class LTQGSolver(object):

    def __init__(self, tqg_params):
        
        self.gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))
        self.grad_x = lambda _u : as_vector((_u.dx(0), 0))
        
        fem_params   = tqg_params.TQG_fem_params
        model_params = tqg_params.TQG_model_params

        self.time_length = model_params.time_length

        self.initial_cond, self.initial_b, self.bathymetry, self.rotation  = tqg_params.set_initial_conditions()

        self.Vdg = fem_params.Vdg
        Vdg      = fem_params.Vdg
        Vcg      = fem_params.Vcg
        self.Vu  = fem_params.Vu

        self.psi0 = Function(Vcg)  # Streamfunctions for different time steps

        self.dq1  = Function(Vdg)  # PV fields for different time steps  # next time step
        self.q1   = Function(Vdg)
        
        self.db1 = Function(Vdg)  # b fields for different time steps   # next time step
        # self.db2 = Function(Vdg)
        self.b1  = Function(Vdg)

        self.Dt = fem_params.dt
        dt      = Constant(self.Dt)

        psi = TrialFunction(Vcg)
        phi = TestFunction(Vcg)

        # --- Elliptic equation ---
        # Build the weak form for the inversion
        # stream function is 0 on the boundary - top and bottom, left and right
        Apsi = (dot(grad(phi), grad(psi)) + phi * psi) * dx
        Lpsi = (self.rotation - self.q1) * phi * dx 

        self.psi_problem = LinearVariationalProblem(Apsi, Lpsi, self.psi0, model_params.bcs)
        self.psi_solver = LinearVariationalSolver(self.psi_problem, solver_parameters={
            'ksp_type':'cg',
            'pc_type':'sor'
        })


        un = 0.5 * (dot(self.gradperp(self.psi0), fem_params.facet_normal) +
                    abs(dot(self.gradperp(self.psi0), fem_params.facet_normal)))

        # un_h = 0.5 * (dot( self.gradperp( 0.5*self.bathymetry - self.psi0), fem_params.facet_normal) +
        #             abs(dot(self.gradperp( 0.5*self.bathymetry - self.psi0), fem_params.facet_normal)))

        un_h = 0.5 * (dot( self.gradperp( 0.5*self.bathymetry), fem_params.facet_normal) +
                    abs(dot(self.gradperp( 0.5*self.bathymetry), fem_params.facet_normal)))

        # --- b equation -----
        b = TrialFunction(Vdg)
        p = TestFunction(Vdg)
        a_mass_b = p * b * dx
        a_int_b = (dot(grad(p), -self.gradperp(self.psi0) * b)) * dx

        a_flux_b = (dot(jump(p), un('+') * b('+') - un('-') * b('-'))) * dS

        arhs_b = a_mass_b - dt * (a_int_b + a_flux_b)
        self.b_problem = LinearVariationalProblem(a_mass_b, action(arhs_b, self.b1), self.db1)  # solve for db1
        self.b_solver = LinearVariationalSolver(self.b_problem, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        })

        # --- q equation -----
        q = TrialFunction(Vdg)
        p_ = TestFunction(Vdg)
        a_mass_ = p_ * q * dx
        a_int_ = ( dot(grad(p_), - self.gradperp(self.psi0)* (q - self.b1) 
                                    - 0.5 * self.gradperp(self.bathymetry)* self.b1)) * dx
                                # + p_*Constant(10.)*q ) * dx

        # a_flux_ = (dot(jump(p_), un('+') * q('+') - un('-') * q('-')  
        #                             + un_h('+') * self.b1('+') - un_h('-') * self.b1('-') )) * dS
        
        a_flux_ = (dot(jump(p_), un('+') * (q('+') - self.b1('+')) - un('-') * (q('-') - self.b1('-'))
                                    + un_h('+') * self.b1('+') - un_h('-') * self.b1('-') )) * dS

        # a_int_ = ( dot(grad(p_), - self.gradperp(self.psi0)* (q - self.db2) 
        #                             - 0.5 * self.gradperp(self.bathymetry)* self.db2 ) ) * dx
        # a_flux_ = (dot(jump(p_), un('+') * q('+') - un('-') * q('-')  
        #                             + un_h('+') * self.db2('+') - un_h('-') * self.db2('-') )) * dS

        arhs_ = a_mass_ - dt * (a_int_ + a_flux_ )

        self.q_problem = LinearVariationalProblem(a_mass_, action(arhs_, self.q1), self.dq1)  # solve for dq1
        self.q_solver = LinearVariationalSolver(self.q_problem, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        })


    def solve(self, dumpfreq, output_name):
        """
        solve for the whole TQG system given initial condition q0

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
        b0 = Function(self.Vdg)
        q0.assign(self.initial_cond)
        b0.assign(self.initial_b)
        
        ssh = Function(self.Vdg, name="SSH")  # sea surface height = psi - 0.5 b1

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")
        b0.rename("Buoyancy")
        v = Function(Vu, name="Velocity")
        
        self.q1.assign(q0)
        self.psi_solver.solve()
        v.project(self.gradperp(self.psi0))
        ssh.assign(Function(self.Vdg).project(self.psi0) - 0.5 * b0)
        
        output_file = File(output_name + ".pvd")
        output_file.write(q0, self.psi0, v, b0, ssh, time=0)
        
        t = 0.
        T = self.time_length
        tdump = 0

        index = 0
        energy = np.zeros(int(T / Dt / dumpfreq)+1)
        kinetic_energy = np.zeros(int(T / Dt / dumpfreq)+1)
        potential_energy = np.zeros(int(T / Dt / dumpfreq)+1)
        
        energy[index] = tqg_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
        kinetic_energy[index] = tqg_kinetic_energy(q0, self.psi0, self.rotation)
        potential_energy[index] = tqg_potential_energy(self.bathymetry, b0)

        # start_time = time.time()
        while t < (T - Dt / 2):

            # Compute the streamfunction for the known value of q0
            self.b1.assign(b0)
            self.q1.assign(q0)
            self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(1)
            self.b1.assign(self.db1)
            self.q1.assign(self.dq1)
            self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(2)
            self.b1.assign(0.75 * b0 + 0.25 * self.db1)
            self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
            self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()

            # Find new solution q^(n+1)
            b0.assign(b0 / 3 + 2 * self.db1 / 3)
            q0.assign(q0 / 3 + 2 * self.dq1 / 3)

            # Store solutions to xml and pvd
            t += Dt
            tdump += 1
            if tdump == dumpfreq:
                tdump -= dumpfreq
                _t = round(t, 2)

                v.project(self.gradperp(self.psi0))
                ssh.assign(Function(self.Vdg).project(self.psi0) - 0.5 * b0)
                output_file.write(q0, self.psi0, v, b0, ssh, time=_t)
                index += 1
                energy[index] = tqg_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
                kinetic_energy[index] = tqg_kinetic_energy(q0, self.psi0, self.rotation)
                potential_energy[index] = tqg_potential_energy(self.bathymetry, b0)
                print(_t)

        self.initial_b.assign(b0)
        self.initial_cond.assign(q0)

        np.save(output_name + "energy", energy)
        np.save(output_name + "kinetic_energy", kinetic_energy)
        np.save(output_name + "potential_energy", potential_energy)
