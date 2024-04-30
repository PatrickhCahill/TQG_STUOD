#
#   Author: Wei Pan
#   Copyright   2017
#
#   particle.py
#
#   code for Particle class
#
import os, sys
sys.path.insert(0, os.path.abspath("../"))
import warnings

import numpy as np
from firedrake import *

from utility import get_pde_filename, get_pde_dir
from utilitiesFiredrake import load_chk_point, fine_msh, coarse_average_fine, get_coarse_mesh
import euler
from math import floor


class Particle(euler.EulerParams, euler.EulerSolver):
    """ 
    Class for a single particle 

    A single particle is an automatically perturbed EulerParams object
    """

    def __init__(self, _time_interval, _dt, _msh, _number, _eof_dir, _var):
        """
        Initialise each ensemble member by constructing a random
        streamfunction (smoothed Gaussian random field) which we 
        then use to advect the truth for a few time steps
        
        :param _time_interval:
        :param _dt:
        :param _msh:
        :param _q: initial condition
        :param _number: particle id
        :param _eof_dir: directory of the eofs
        :param _var: variance captured by the number of eofs
        """
        # start_time = time.time()
        euler.EulerParams.__init__(self, _time_interval, _dt, _msh)
        euler.EulerSolver.__init__(self, self)
        self.id = _number
        # self.initial_cond.assign(_q)
        self.eof_dir = _eof_dir
        self.eof_var = _var

    def output_for_visual(self, _particle_name_prefix, _output_name, _num_time_steps, _t0=0, _dt=2.5):
        warnings.warn("Deprecated", DeprecationWarning)
        """
        read in particle pv h5 file and get the streamfunction and velocity for visual output
        :param _particle_name_prefix: dir + particle name and id
        :param _output_name: dir + output name
        :param _num_time_steps:
        :param _t0:
        :return:
        """
        output_file = File(_output_name + ".pvd")
        for t in range(_num_time_steps):
            name = _particle_name_prefix + "_{}".format(t)
            with DumbCheckpoint(name, mode=FILE_READ) as chk:
                chk.load(self.q1, "Vorticity")
            self.q1.rename("Vorticity")
            self.psi0.rename("Streamfunction")
            self.psi_solver.solve()
            v = Function(self.Vu, name="Velocity")
            v.project(self.gradperp(self.psi0))
            output_file.write(self.q1, self.psi0, v, time=_t0+t*_dt)    # output every ett

    def project_output_fine_for_visual(self, _res, output_name):
        """
        function for projecting fine solution outputs to coarse resolution
        """
        q_f = Function(FunctionSpace(fine_msh, "DG", 1))
        total = 830
        t = 365.
        output_file = File(get_pde_dir(output_name + '.pvd'))
        # energy = []
        for num in range(730, total, 5):
            print(t)
            filename_q_f = get_pde_filename(num)  #output_directory("q_" + str(num), '/PDEsolution')
            chk_q = DumbCheckpoint(filename_q_f, mode=FILE_READ)
            chk_q.load(q_f, name="Vorticity")
            coarse_msh = get_coarse_mesh(_res)
            q_c = coarse_average_fine(q_f, coarse_msh, FunctionSpace(coarse_msh, "DG", 1), _res*_res)
            self.q1.assign(q_c)
            self.psi_solver.solve()
            q_c.rename("Vorticity")
            self.psi0.rename("Streamfunction")
            v = Function(self.Vu, name="Velocity")
            v.project(self.gradperp(self.psi0))
            output_file.write(q_c, self.psi0, v, time=t)
            # energy.append(norm(v)**2 * 0.5)
            t += 2.5
        # np.savetxt(utility.output_directory("fine_projected_energy.csv"), np.array(energy), delimiter=",")


    def random_ensemble_member_generator_direct_perturbation(self, _q0, _k_sqr, _output_name, _chkpt_flag=True):
        """
        directly perturb the initial condition

        :param _q0:
        :param _k_sqr:
        :param _output_name:
        :param _chkpt_flag:
        :return:
        """

        q0 = Function(self.euler_fem_params.Vdg)
        q0.assign(_q0)

        q0.rename("Vorticity")

        # draw a random number
        np.random.seed(None)
        noise = np.random.normal(0., 2, q0.dat.data[:].shape)

        with DumbCheckpoint(_output_name, mode=FILE_CREATE, single_file=False) as chk:
            q0.dat.data[:] += noise
            chk.store(q0)

        # Dt = self.Dt
        #
        # q0 = Function(self.euler_fem_params.Vdg)
        # q0.assign(_q0)
        #
        # q0.rename("Vorticity")
        # self.psi0.rename("Streamfunction")
        #
        # # draw a random number
        # np.random.seed(None)
        # psi = Function(self.euler_fem_params.Vcg)  # randomly generated psi
        # self.q1.project(q0)
        # self.psi_solver.solve()
        # noise = np.random.normal(0., 0.5, psi.dat.data[:].shape)
        # # rescale_array(noise, np.amin(self.psi0.dat.data[:]), np.amax(self.psi0.dat.data[:]))
        # psi.dat.data[:] += noise
        # psi.dat.data[:] = self.psi0.dat.data[:] + self.helmholtz_solver(psi, 1.).dat.data[:]
        # self.psi0.assign(psi)
        # self.solve_for_q_given_psi(self.psi0, q0)
        #
        # with DumbCheckpoint(_output_name, mode=FILE_CREATE, single_file=False) as chk:
        #     chk.store(q0)


    def random_ensemble_member_generator(self, _q0, _k_sqr, _output_name, _chkpt_flag=True):
        """
        DEFORMATION PROCEDURE

        randomly generates a streamfunction which is constant on the boundary, this gives a div free u
        use this u to perturb a given initial q0

        :param _q0:
        :param _output_name:
        :param _chkpt_flag:
        :return:
        """
        Dt = self.Dt

        q0 = Function(self.euler_fem_params.Vdg)
        q0.assign(_q0)

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")

        # draw a random number
        np.random.seed(None)
        num_random_combinations = 1
        psi = Function(self.euler_fem_params.Vcg)   # randomly generated psi
        weights = np.random.normal(0., .25, num_random_combinations) # beta
        for num in range(num_random_combinations):
            q_fine = Function(FunctionSpace(fine_msh, "DG", 1))
            q_index = np.random.randint(0, 729)     # tau
            load_chk_point(get_pde_filename(q_index), q_fine, "Vorticity")
            # load_chk_point(output_directory('q_{}'.format(q_index), '/PDEsolution'), q_fine, "Vorticity")
            self.q1.project(coarse_average_fine(q_fine, q0.function_space().mesh(), q0.function_space(), _k_sqr))
            self.psi_solver.solve()
            psi.assign(assemble(psi + weights[num]*self.psi0))
        self.psi0.assign(psi)

        un = 0.5 * (dot(self.gradperp(self.psi0), self.euler_fem_params.facet_normal) +
                    abs(dot(self.gradperp(self.psi0), self.euler_fem_params.facet_normal)))

        # Now the variational problem for the advection equation itself. ::
        q = TrialFunction(self.Vdg)
        p = TestFunction(self.Vdg)
        dt = Constant(self.Dt)
        a_mass = p * q * dx

        a_int = (dot(grad(p), -self.gradperp(self.psi0) * q)) * dx

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

        with DumbCheckpoint(_output_name, mode=FILE_CREATE, single_file=False) as chk:
            t = 0.
            T = self.euler_model_params.time_length

            while t < (T - Dt / 2):
                self.q1.assign(q0)
                self.q_solver.solve()

                # Find intermediate solution q^(1)
                self.q1.assign(self.dq1)
                self.q_solver.solve()

                # Find intermediate solution q^(2)
                self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
                self.q_solver.solve()

                # Find new solution q^(n+1)
                q0.assign(q0 / 3 + 2 * self.dq1 / 3)

                # Store solutions to xml and pvd
                t += Dt
            chk.store(q0)

    def spdesolver(self, _q0, _output_name, _chkpt_flag=False, **kwargs):
        """
        SPDE solver

        :param :
        :kwargs proposal_step: Rho in the MCMC move
        :kwargs state_store: Data store for storing the BM path. This data store covers the whole ensemble thus works with multiprocessing.
        :return:
            :noise: the bm path used for calculating the solution
        """
        Vu = self.Vu
        Dt = self.Dt

        q0 = Function(self.euler_fem_params.Vdg)
        q0.assign(_q0)  # so it doesn't change q0

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")
        v = Function(Vu, name="Velocity")
        v.project(self.gradperp(self.psi0))
        t = 0.
        T = self.euler_model_params.time_length

        from math import ceil
        iter_steps = ceil(T / Dt - 0.5)  # number of while loop iteration

        # zetas.shape[0] is the number of EOFs
        zetas = np.genfromtxt("{}zetas_{}.csv".format(self.eof_dir, self.eof_var), delimiter=",")
        noise = np.zeros(zetas.shape[0] * iter_steps)

        np.random.seed(None)

        # random_state = kwargs.get('random_state')
        rho = kwargs.get('proposal_step')
        state_store = kwargs.get('state_store')

        if 'proposal_step' in kwargs and 'state_store' in kwargs:
            # np.random.set_state(random_state)
            # noise = rho*np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0]*iter_steps)
            # generate mcmc move bm increments
            # np.random.seed(None)  # so to reset the seed for the new bm
            noise += rho * state_store + np.sqrt((1. - rho ** 2) / Dt) * np.random.normal(0., 1., zetas.shape[
                0] * iter_steps)
        # elif state_store:
        # random_state is not supplied, but state_store is, so we need to store the current state
        # state_store[self.id] = np.random.get_state()
        # noise = np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0]*iter_steps)
        else:
            # when neither additional parameters are supplied, so we just generate bm increments without storing
            # random number generator state
            noise += np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0] * iter_steps)

        step = 0
        while t < (T - Dt / 2):
            # generate BM for each EOF
            # due to dt scaling the FEM setup, we need to scale by 1/sqrt(dt) to get the correct BM variance
            # bms = np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0])
            bms = noise[step:step + zetas.shape[0]]
            psi0_perturbation = np.sum((zetas.T * bms).T, axis=0)
            step += zetas.shape[0]

            # Compute the streamfunction for the known value of q0
            self.q1.assign(q0)
            self.psi_solver.solve()
            self.psi0.dat.data[:] += psi0_perturbation
            self.q_solver.solve()

            # Find intermediate solution q^(1)
            self.q1.assign(self.dq1)
            self.psi_solver.solve()
            self.psi0.dat.data[:] += psi0_perturbation
            self.q_solver.solve()

            # Find intermediate solution q^(2)
            self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
            self.psi_solver.solve()
            self.psi0.dat.data[:] += psi0_perturbation
            self.q_solver.solve()

            # Find new solution q^(n+1)
            q0.assign(q0 / 3 + 2 * self.dq1 / 3)

            # Store solutions to xml and pvd
            t += Dt

        self.initial_cond.assign(q0)  # save for norm comparison

        if _chkpt_flag:
            with DumbCheckpoint(_output_name, mode=FILE_CREATE, single_file=True) as chk:
                chk.store(q0)

        return noise


    def solver(self, _dumpfreq, _q0, _output_name, _output_visual_flag=False, _chkpt_flag=False, **kwargs):
        """
        SPDE solver

        :param :
        :kwargs proposal_step: Rho in the MCMC move
        :kwargs state_store: Data store for storing the BM path. This data store covers the whole ensemble thus works with multiprocessing.
        :return:
            :noise: the bm path used for calculating the solution
        """
        warnings.warn("This function is no longer required. It uses _dumpfreq, this feature is removed", DeprecationWarning)

        Vu = self.Vu
        Dt = self.Dt

        q0 = Function(self.euler_fem_params.Vdg)
        q0.assign(_q0)  # so it doesn't change q0

        q0.rename("Vorticity")
        self.psi0.rename("Streamfunction")
        v = Function(Vu, name="Velocity")
        v.project(self.gradperp(self.psi0))
        t = 0.
        T = self.euler_model_params.time_length
        tdump = 0

        from math import ceil
        iter_steps = ceil(T / Dt - 0.5)  # number of while loop iteration

        # zetas.shape[0] is the number of EOFs
        zetas = np.genfromtxt("{}zetas_{}.csv".format(self.eof_dir, self.eof_var), delimiter=",")
        noise = np.zeros(zetas.shape[0]*iter_steps)

        np.random.seed(None)
        # vis_data = kwargs.get('vis_data')

        output_file = File(_output_name + ".pvd")

        with DumbCheckpoint(_output_name, mode=FILE_CREATE, single_file=False) as chk:
            if _output_visual_flag:
                output_file.write(q0, self.psi0, v)

            if _chkpt_flag:
                chk.store(q0)

            # if vis_data:
            #     # store_step = floor(round(t / Dt / _dumpfreq))
            #     # print(store_step, t, _dumpfreq)
            #     vis_data.store_data(self.id, q0, 0)

            # random_state = kwargs.get('random_state')
            rho = kwargs.get('proposal_step')
            state_store = kwargs.get('state_store')

            if 'proposal_step' in kwargs and 'state_store' in kwargs:
                # np.random.set_state(random_state)
                # noise = rho*np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0]*iter_steps)
                # generate mcmc move bm increments
                # np.random.seed(None)  # so to reset the seed for the new bm
                noise += rho * state_store + np.sqrt((1.-rho**2)/Dt) * np.random.normal(0., 1., zetas.shape[0]*iter_steps)
            # elif state_store:
                # random_state is not supplied, but state_store is, so we need to store the current state
                # state_store[self.id] = np.random.get_state()
                # noise = np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0]*iter_steps)
            else:
                # when neither additional parameters are supplied, so we just generate bm increments without storing
                # random number generator state
                noise += np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0]*iter_steps)

            step = 0
            while t < (T - Dt / 2):
                # generate BM for each EOF
                # due to dt scaling the FEM setup, we need to scale by 1/sqrt(dt) to get the correct BM variance
                # bms = np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0])
                bms = noise[step:step+zetas.shape[0]]
                psi0_perturbation = np.sum((zetas.T * bms).T, axis=0)
                step += zetas.shape[0]

                # Compute the streamfunction for the known value of q0
                self.q1.assign(q0)
                self.psi_solver.solve()
                self.psi0.dat.data[:] += psi0_perturbation
                self.q_solver.solve()

                # Find intermediate solution q^(1)
                self.q1.assign(self.dq1)
                self.psi_solver.solve()
                self.psi0.dat.data[:] += psi0_perturbation
                self.q_solver.solve()

                # Find intermediate solution q^(2)
                self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
                self.psi_solver.solve()
                self.psi0.dat.data[:] += psi0_perturbation
                self.q_solver.solve()

                # Find new solution q^(n+1)
                q0.assign(q0 / 3 + 2 * self.dq1 / 3)

                # Store solutions to xml and pvd
                t += Dt
                tdump += 1
                if tdump == _dumpfreq:
                    tdump -= _dumpfreq
                    _t = round(t, 2)

                    # if vis_data:
                    #     store_step = floor(round(t /Dt / _dumpfreq))
                    #     # print(store_step, t, _dumpfreq)
                    #     vis_data.store_data(self.id, q0, store_step)

                    if _output_visual_flag:
                        print(_t)
                        self.q1.assign(q0)
                        self.psi_solver.solve()
                        v.project(self.gradperp(self.psi0))
                        output_file.write(q0, self.psi0, v, time=self.base_time+_t)

                    if _chkpt_flag:
                        chk.new_file()
                        chk.store(q0)

            self.initial_cond.assign(q0)    # save for norm comparison

        return noise

    # def l2_error(self, fine_scale_value):
    #     """
    #     this produces the L2 norm of the difference between fine_scale_value and q0
    #
    #     :param fine_scale_value:
    #     :return:
    #     """
    #     fs = self.initial_cond.function_space()
    #     uf_proj = Function(fs)
    #     inject(fine_scale_value, uf_proj)
    #     return norm(assemble(self.initial_cond - uf_proj))
    #
    # def l2_error_q1_q2(self, q1, q2):
    #     """
    #     this produces the L2 norm of the difference between fine_scale_value and q0
    #
    #     :param fine_scale_value:
    #     :return:
    #     """
    #     fs = self.initial_cond.function_space()
    #     uf_proj = Function(fs)
    #     inject(q2, uf_proj)
    #     return norm(assemble(q1 - uf_proj))
