#
# Author Wei Pan
# 
# STQG solver
#

from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
#from tqg.solver import TQGSolver
from tqg.diagnostics import tqg_mesh_grid
from tqg.diagnostics import tqg_energy, tqg_kinetic_energy, tqg_potential_energy 


class STQGSolver(object):

    def __init__(self, tqg_params, bathymetry_xi_scaling=1.):
        self.id = 0

        #TQGSolver.__init__(self, tqg_params)
        self.gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))
        
        fem_params   = tqg_params.TQG_fem_params
        model_params = tqg_params.TQG_model_params

        self.time_length = model_params.time_length

        #self.bathymetry is a CG function
        self.initial_cond, self.initial_b, self.bathymetry, self.rotation  = tqg_params.set_initial_conditions()

        forcing = tqg_params.set_forcing() 
        damping_rate = tqg_params.set_damping_rate()

        self.bathymetry_xi_scaling = bathymetry_xi_scaling

        self.Vdg = fem_params.Vdg
        Vdg      = fem_params.Vdg
        self.Vcg = fem_params.Vcg
        Vcg      = fem_params.Vcg
        self.Vu  = fem_params.Vu

        self.psi0 = Function(Vcg)  
        self.ssh  = Function(Vcg, name="SSH")

        self.dq1  = Function(Vdg)  
        self.q1   = Function(Vdg)
        
        self.db1 = Function(Vdg)  
        #self.db2 = Function(Vdg)
        self.db2 = Function(Vcg)
        self.b1  = Function(Vdg)

        self.Dt = fem_params.dt
        dt      = Constant(self.Dt)

        psi = TrialFunction(Vcg)
        phi = TestFunction(Vcg)

        # --- elliptic equation ---
        Apsi = (dot(grad(phi), grad(psi)) + phi * psi) * dx
        Lpsi = (self.rotation - self.q1) * phi * dx 

        psi_problem = LinearVariationalProblem(Apsi, Lpsi, self.psi0, model_params.bcs)
        self.psi_solver = LinearVariationalSolver(psi_problem, \
            solver_parameters={
                'ksp_type':'cg',
                'pc_type':'sor'
            }
        )

        # --- b equation -----
        un = 0.5 * (dot(self.gradperp(self.psi0), fem_params.facet_normal) \
            + abs(dot(self.gradperp(self.psi0), fem_params.facet_normal)))

        _un_ = dot(self.gradperp(self.psi0), fem_params.facet_normal)
        _abs_un_ = abs(_un_)

        b = TrialFunction(Vdg)
        p = TestFunction(Vdg)
        a_mass_b = p * b * dx
        #a_int_b = (dot(grad(p), -self.gradperp(self.psi0) * b)) * dx
        a_int_b = (dot(grad(p), -self.gradperp(self.psi0) * b)
                    - p * (forcing - b * damping_rate) ) *dx

        # a_flux_b = (p('+') * _un_('+') + p('-') * _un_('-') ) * avg(b) * dS  # central flux
        # a_flux_b = (dot(jump(p), un('+')*b('+') - un('-')*b('-')) )*dS  # upwinding
        a_flux_b =  0.5*jump(p)*(2*_un_('+')*avg(b) + _abs_un_('+')*jump(b))*dS  

        arhs_b = a_mass_b - dt * (a_int_b + a_flux_b)

        b_problem = LinearVariationalProblem(a_mass_b, action(arhs_b, self.b1), self.db1 \
                 , bcs=model_params.dgbcs)  # solve for db1
        self.b_solver = LinearVariationalSolver(b_problem, \
            solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
            }
        )

        # --- q equation -----
        # _uhn_ = dot( self.gradperp(self.bathymetry), fem_params.facet_normal)
        # _abs_uhn_ = abs(_uhn_)

        q = TrialFunction(Vdg)
        p_ = TestFunction(Vdg)
        a_mass_ = p_ * q * dx

        # works!!! # 
        a_int_ = ( dot(grad(p_), -self.gradperp(self.psi0)*(q - self.db1))
                  + p_ * div(self.db2 * self.gradperp(0.5*self.bathymetry))) *dx
                 #- p_ * (forcing - q * damping_rate) ) *dx

        # below test case by fixing b in the grad b forcing term
        # a_int_ = ( dot(grad(p_), -self.gradperp(self.psi0)*(q - self.db2)) \
        #          + p_ * div(self.initial_cond* self.gradperp(0.5*self.bathymetry)) ) *dx

        # original unstable case
        # a_flux_ = (dot(jump(p_), un('+') * (q('+') - self.db2('+')) - un('-') * (q('-') - self.db2('-'))
        #                         + un_h('+') * self.db2('+') - un_h('-') * self.db2('-') 
        #                         )) * dS
        a_flux_ =  0.5*jump(p_)*(2*_un_('+')*avg(q-self.db1) + _abs_un_('+')*jump(q - self.db1))*dS  
        # a_flux_ = (dot(jump(p_), un('+') * q('+') - un('-') * q('-')))*dS

        arhs_ = a_mass_ - dt * (a_int_ + a_flux_ )

        q_problem = LinearVariationalProblem(a_mass_, action(arhs_, self.q1), self.dq1)  # solve for dq1

        self.q_solver = LinearVariationalSolver(q_problem, 
            solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
            }
        )
        
        # ----- vertex based limiter -------
        # self.vb_limiter = VertexBasedLimiter(Vdg)

        self.x = tqg_params.TQG_fem_params.x
        self.Vcg = tqg_params.TQG_fem_params.Vcg
        self.solver_name = 'STQG solver'

    def load_initial_conditions_from_file(self, h5_data_name, comm=None):
        """
        Read in initial conditions from saved (checkpointed) data file

        Loads pv, buoyancy and streamfunction
        """
        # PETSc.Sys.Print(norm(self.initial_cond))
        with DumbCheckpoint(h5_data_name, mode=FILE_READ, comm=comm) as chk:
            chk.load(self.initial_cond, name="PotentialVorticity")
            chk.load(self.initial_b, name="Buoyancy")
            chk.load(self.psi0, name="Streamfunction")
            try:
                chk.load(self.ssh, name="SSH")
            except:
                print('no ssh in data')
                #pass
        # PETSc.Sys.Print(norm(self.initial_cond))
        
    
    def save_velocity_grid_data(self, h5_data_name, res):
        """
        Takes checkpoint data (pv, buoyancy) and save corresponding velocity field grid values for spectral analysis
        h5_data_name must end in '.h5'
        """
        mesh_grid = tqg_mesh_grid(res)
        q0 = Function(self.Vdg)
        with DumbCheckpoint(h5_data_name, mode=FILE_READ) as chk:
            chk.load(q0, name="PotentialVorticity")
        v = Function(VectorFunctionSpace(self.Vdg.mesh(),"CG", 1), name="Velocity")
        self.q1.assign(q0)
        self.psi_solver.solve()
        v.project(self.gradperp(self.psi0))
        # PETSc.Sys.Print(norm(v), v.at([0.5, 0.5], tolerance=1e-10), flush=True)
        # PETSc.Sys.Print(h5_data_name)
        np.save(h5_data_name, np.asarray(v.at(mesh_grid, tolerance=1e-10))) 


    def solve_for_streamfunction_data_from_file(self, h5_data_name):
        """
        Load h5 data, and solve for stream function
        """
        self.load_initial_conditions_from_file(h5_data_name)
        self.q1.assign(self.initial_cond)
        self.psi_solver.solve()


    def get_streamfunction_grid_data(self, h5_data_name, grid_point):
        """
        Takes checkpoint data (pv, buoyancy) and save corresponding streamfunction grid values for autocorrelation analysis
        h5_data_name must end in '.h5'
        """
        q0 = Function(self.Vdg)
        with DumbCheckpoint(h5_data_name, mode=FILE_READ) as chk:
            chk.load(q0, name="PotentialVorticity")
        self.q1.assign(q0)
        self.psi_solver.solve()

        return np.asarray(self.psi0.at(grid_point, tolerance=1e-10)) 


    #def solve(self, dumpfreq, output_name, data_output_name, ensemble_comm, do_save_data=False, do_save_visual=True, do_save_spectrum=False, res=0, zetas_file_name=None, **kwargs):
    def solve(self, dumpfreq, output_name, data_output_name, comm_manager, do_save_data=False, do_save_visual=True, do_save_spectrum=False, res=0, zetas_file_name=None, xi_scaling=1, bathymetry_xi=False, **kwargs):
        """
        solve the STQG system given initial condition q0

        :param dumpfreq:
        :param _q0:
        :param output_name: name of output files, stored in utility.output_directory
        :param output_visual_flag: if True, this function will output pvd files for visualisation at a frequency
        defined by dumpfreq
        :param chkpt_flag: if True, this function will store solved q as chkpoint file at solution times defined by
         dumpfreq
        :param zetas_file_name: numpy file name
        :return: 
        """
        #PETSc.Sys.Print(self.solver_name,flush=True)
        
        # assume q0 is given in initial_cond
        q0 = Function(self.Vdg)
        b0 = Function(self.Vdg)
        q0.assign(self.initial_cond)
        b0.assign(self.initial_b)
        
        #ssh = Function(self.Vcg, name="SSH")  # sea surface height = psi - 0.5 b1

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("PotentialVorticity")
        self.psi0.rename("Streamfunction")
        b0.rename("Buoyancy")
        v = Function(Vu, name="Velocity")
        
        self.q1.assign(q0)
        self.psi_solver.solve()
        v.project(self.gradperp(self.psi0))
        #ssh.assign(Function(self.Vdg).project(self.psi0) - 0.5 * b0)
        self.ssh.assign(self.psi0 - 0.5 * Function(self.Vcg).project(b0))

        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm = comm_manager.comm
        
        # ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
        procno = ensemble_comm.rank
        
        output_file = None
        data_chk = None
        
        compute_kinetic_energy = lambda _w, _psi, _f : assemble( -0.5 * _psi * (_w - _f ) *dx )
        compute_potential_energy = lambda _h, _b : assemble( -0.5* _h * _b * dx)
        compute_total_energy = lambda _w, _psi, _f, _h, _b : assemble( -0.5 * (_psi*(_w-_f) + _h*_b ) *dx ) 
        compute_casimir = lambda _w, _b : assemble( (_b + _w*_b)*dx )
        compute_non_casimir = lambda _w, _b : assemble( (_b + _w*_w*_b) *dx )

        kinetic_energy_series = []
        potential_energy_series = []
        total_energy_series = []
        casimir_series = []
        non_casimir_series = []

        if (procno == 0):
            if self.solver_name == 'TQG solver':
                _ke = compute_kinetic_energy(q0, self.psi0, self.rotation)
                _pe = compute_potential_energy(self.bathymetry, b0)
                _te = compute_total_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
                kinetic_energy_series.append(_ke)
                potential_energy_series.append(_pe)
                total_energy_series.append(_te)
                casimir_series.append(compute_casimir(q0, b0))
                non_casimir_series.append(compute_non_casimir(q0, b0))
                
                # print(_te, _ke, _pe, flush=True)

            if do_save_visual:
                print("saving visual ," ,ensemble_comm.rank, flush=True)
                #output_file = File(output_name + ".pvd")
                output_file = File(output_name + ".pvd", comm=spatial_comm)
                output_file.write(q0, self.psi0, v, b0, self.ssh, time=0)
            if do_save_data:
                data_chk = DumbCheckpoint(data_output_name, single_file=False, mode=FILE_CREATE, comm=spatial_comm)
                data_chk.store(q0)
                data_chk.store(b0)
                data_chk.store(self.psi0)
                data_chk.store(self.ssh)
        
        t = 0.
        T = self.time_length
        tdump = 0

        # index = 0
        # energy = np.zeros(int(T / Dt / dumpfreq)+1)
        # kinetic_energy = np.zeros(int(T / Dt / dumpfreq)+1)
        # potential_energy = np.zeros(int(T / Dt / dumpfreq)+1)
        
        # energy[index] = tqg_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
        # kinetic_energy[index] = tqg_kinetic_energy(q0, self.psi0, self.rotation)
        # potential_energy[index] = tqg_potential_energy(self.bathymetry, b0)

        mesh_grid = tqg_mesh_grid(res)
        index = 0

        from math import ceil
        iter_steps = ceil(T/Dt - 0.5) # number of while loop iterations
        # zetas are EOFs and should be of the shape psi.dat.data[:]

        np.random.seed(None)
        rho = kwargs.get('proposal_step')
        state_store = kwargs.get('state_store')
        

        zetas = None
        noise = None
        psi0_perturbation = 0 
        bms = None
        if 1: #self.solver_name == 'STQG solver':
            #print(zetas_file_name)
            #zetas = xi_scaling * np.load(zetas_file_name) if zetas_file_name is not None else  np.asarray([Function(self.Vcg).dat.data])

            if (zetas_file_name is not None) and (bathymetry_xi is False):
                print('zeta is file!!!!!!!!!!', flush=True)
                zetas = xi_scaling * np.load(zetas_file_name)
            else:
                if bathymetry_xi is True:
                    print('zetas = bathymetry', flush=True)
                    zetas = xi_scaling * np.asarray([self.bathymetry.dat.data]) 
                else:
                    print('no zeta', flush=True)
                    zetas = np.asarray([Function(self.Vcg).dat.data])
            #print(zetas.shape)
            #Function(self.Vcg)
            #zetas = np.asarray([zeta.dat.data])  

            if 'proposal_step' in kwargs and 'state_store' in kwargs:
                noise = rho * state_store + np.sqrt((1. - rho**2) /Dt) * np.random.normal(0, 1, zetas.shape[0] * iter_steps)
                
            else:
                noise = np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0] * iter_steps) if ((zetas_file_name is not None) or (bathymetry_xi is True)) else np.zeros(zetas.shape[0] *  iter_steps)
                print(noise, flush=True)
            #print(noise.shape, np.asarray(self.psi0.dat.data).shape)

        step = 0
        while t < (T - Dt / 2):
            # sort out BM
            if 1: #self.solver_name == 'STQG solver':
                bms = noise[step:step+zetas.shape[0]]
                step += zetas.shape[0]

            # Compute the streamfunction for the known value of q0
            self.b1.assign(b0)
            self.q1.assign(q0)
            self.db2.project(b0)

            self.psi_solver.solve()
            psi0_perturbation = np.sum((zetas.T * bms).T, axis=0) 
            #print(psi0_perturbation, flush=True)
            
            self.psi0.dat.data[:] += psi0_perturbation
            self.b_solver.solve()
            self.q_solver.solve()     

            # # Find intermediate solution q^(1)
            self.b1.assign(self.db1)
            self.q1.assign(self.dq1)
            self.db2.project(self.b1)
            
            self.psi_solver.solve()
            self.psi0.dat.data[:] += psi0_perturbation
            self.b_solver.solve()
            self.q_solver.solve()

            # # Find intermediate solution q^(2)
            self.b1.assign(0.75 * b0 + 0.25 * self.db1)
            self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
            self.db2.project(self.b1)

            self.psi_solver.solve()
            self.psi0.dat.data[:] += psi0_perturbation
            self.b_solver.solve()
            self.q_solver.solve()

            # # Find new solution q^(n+1)
            b0.assign(b0 / 3 + 2 * self.db1 / 3)
            q0.assign(q0 / 3 + 2 * self.dq1 / 3)

            # Store solutions to xml and pvd
            t += Dt
            tdump += 1
            if tdump == dumpfreq:
                tdump -= dumpfreq
                _t = round(t, 5)
                _ke, _pe, _te = 0, 0, 0 
                if (procno == 0):
                    self.q1.assign(q0)
                    self.psi_solver.solve()
                    v.project(self.gradperp(self.psi0))
                    self.ssh.assign(self.psi0 - 0.5 * Function(self.Vcg).project(b0))

                    if self.solver_name == 'TQG solver':
                        _ke = compute_kinetic_energy(q0, self.psi0, self.rotation)
                        _pe = compute_potential_energy(self.bathymetry, b0)
                        _te = compute_total_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
                        kinetic_energy_series.append(_ke)
                        potential_energy_series.append(_pe)
                        total_energy_series.append(_te)
                        non_casimir_series.append(compute_non_casimir(q0, b0))

                    if do_save_visual:
                        #ssh.assign(Function(self.Vdg).project(self.psi0) - 0.5 * b0)
                        output_file.write(q0, self.psi0, v, b0, self.ssh, time=_t)
                    if do_save_spectrum:                        
                        # -------- FFT --------------------------------------
                        index += 1
                        np.save(data_output_name + "_energy_{}".format(index), np.asarray(v.at(mesh_grid, tolerance=1e-10)))                            
                        # -------- FFT end --------------------------------------
                    if do_save_data:
                        data_chk.new_file()
                        data_chk.store(q0)
                        data_chk.store(b0)
                        data_chk.store(self.psi0) # for calibration
                        data_chk.store(self.ssh)

                PETSc.Sys.Print(_t,"/",T, " ", _te, _ke, _pe, flush=True)
        self.initial_cond.assign(q0)
        self.initial_b.assign(b0)

        if do_save_data and procno==0:
            data_chk.close()

        return noise
        #return np.asarray(kinetic_energy_series), np.asarray(potential_energy_series), np.asarray(total_energy_series), np.asarray(casimir_series), np.asarray(non_casimir_series)
