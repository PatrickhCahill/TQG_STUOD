#
# Author Wei Pan
#
# TQG solver -- removed the Jacobian term from the pv equation
#

from firedrake import *
import numpy as np
from tqg.diagnostics import tqg_energy, tqg_kinetic_energy, tqg_potential_energy, tqg_mesh_grid


class TQGSolver(object):

    def __init__(self, tqg_params):
        
        self.gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))
        
        fem_params   = tqg_params.TQG_fem_params
        model_params = tqg_params.TQG_model_params

        self.time_length = model_params.time_length

        self.initial_cond, self.initial_b, self.bathymetry, self.rotation  = tqg_params.set_initial_conditions()

        self.Vdg = fem_params.Vdg
        Vdg      = fem_params.Vdg
        Vcg      = fem_params.Vcg
        self.Vu  = fem_params.Vu

        self.psi0 = Function(Vcg)  

        self.dq1  = Function(Vdg)  
        self.q1   = Function(Vdg)
        
        self.db1 = Function(Vdg)  
        self.db2 = Function(Vdg)
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
        a_int_b = (dot(grad(p), -self.gradperp(self.psi0) * b)) * dx

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

        a_int_ = ( dot(grad(p_), -self.gradperp(self.psi0)*(q - self.db2)*0) \
                  + p_ * div(self.db2* self.gradperp(0.5*self.bathymetry)) ) *dx

        # below test case by fixing b in the grad b forcing term
        # a_int_ = ( dot(grad(p_), -self.gradperp(self.psi0)*(q - self.db2)) \
        #          + p_ * div(self.initial_cond* self.gradperp(0.5*self.bathymetry)) ) *dx

        # original unstable case
        # a_flux_ = (dot(jump(p_), un('+') * (q('+') - self.db2('+')) - un('-') * (q('-') - self.db2('-'))
        #                         + un_h('+') * self.db2('+') - un_h('-') * self.db2('-') 
        #                         )) * dS
        a_flux_ =  0 # 0.5*jump(p_)*(2*_un_('+')*avg(q-self.db2) + _abs_un_('+')*jump(q - self.db2)) *dS  
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

    def load_initial_conditions_from_file(self, h5_data_name):
        """
        Read in initial conditions from saved (checkpointed) data file
        """
        # PETSc.Sys.Print(norm(self.initial_cond))
        with DumbCheckpoint(h5_data_name, mode=FILE_READ) as chk:
            chk.load(self.initial_cond, name="PotentialVorticity")
            chk.load(self.initial_b, name="Buoyancy")
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


    def solve(self, dumpfreq, output_name, data_output_name, ensemble_comm, do_save_data=False, do_save_visual=True, do_save_spectrum=False, res=0):
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
        PETSc.Sys.Print('TQG solver',flush=True)
        
        # assume q0 is given in initial_cond
        q0 = Function(self.Vdg)
        b0 = Function(self.Vdg)
        q0.assign(self.initial_cond)
        b0.assign(self.initial_b)
        
        ssh = Function(self.Vdg, name="SSH")  # sea surface height = psi - 0.5 b1

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("PotentialVorticity")
        self.psi0.rename("Streamfunction")
        b0.rename("Buoyancy")
        v = Function(Vu, name="Velocity")
        
        self.q1.assign(q0)
        self.psi_solver.solve()
        v.project(self.gradperp(self.psi0))
        ssh.assign(Function(self.Vdg).project(self.psi0) - 0.5 * b0)
        
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
            kinetic_energy_series.append(compute_kinetic_energy(q0, self.psi0, self.rotation))
            potential_energy_series.append(compute_potential_energy(self.bathymetry, b0))
            total_energy_series.append(compute_total_energy(q0, self.psi0, self.rotation, self.bathymetry, b0))
            casimir_series.append(compute_casimir(q0, b0))
            non_casimir_series.append(compute_non_casimir(q0, b0))

            if do_save_visual:
                output_file = File(output_name + ".pvd")
                output_file.write(q0, self.psi0, v, b0, ssh, time=0)
            if do_save_data:
                data_chk = DumbCheckpoint(data_output_name, single_file=False, mode=FILE_CREATE)
                data_chk.store(q0)
                data_chk.store(b0)
                data_chk.store(self.psi0)

        
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

        # start_time = time.time()
        # b_errors = []
        # q_errors = []
        while t < (T - Dt / 2):
            # print(round(t, 3)) 
            # Compute the streamfunction for the known value of q0
            self.b1.assign(b0)
            self.q1.assign(q0)
            self.db2.assign(b0)
            # self.db2.assign(self.initial_b)  # hardcoding b in rk3 step for q

            self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()     

            # # Find intermediate solution q^(1)
            self.b1.assign(self.db1)
            self.q1.assign(self.dq1)
            self.db2.assign(self.b1)
            
            self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()

            # # Find intermediate solution q^(2)
            self.b1.assign(0.75 * b0 + 0.25 * self.db1)
            self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
            self.db2.assign(self.b1)

            self.psi_solver.solve()
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
                _t = round(t, 4)
                if (procno == 0):
                    self.q1.assign(q0)
                    self.psi_solver.solve()
                    v.project(self.gradperp(self.psi0))
                    kinetic_energy_series.append(compute_kinetic_energy(q0, self.psi0, self.rotation))
                    potential_energy_series.append(compute_potential_energy(self.bathymetry, b0))
                    total_energy_series.append(compute_total_energy(q0, self.psi0, self.rotation, self.bathymetry, b0))
                    casimir_series.append(compute_casimir(q0, b0))
                    non_casimir_series.append(compute_non_casimir(q0, b0))

                    if do_save_visual:
                        ssh.assign(Function(self.Vdg).project(self.psi0) - 0.5 * b0)
                        output_file.write(q0, self.psi0, v, b0, ssh, time=_t)
                    if do_save_spectrum:                        
                        # -------- FFT --------------------------------------
                        index += 1
                        np.save(data_output_name + "_energy_{}".format(index), np.asarray(v.at(mesh_grid, tolerance=1e-10)))                            
                        # -------- FFT end --------------------------------------
                    if do_save_data:
                        data_chk.new_file()
                        data_chk.store(q0)
                        data_chk.store(b0)
                        data_chk.store(self.psi0)  # for calibration
                PETSc.Sys.Print(_t,"/",T,flush=True)



        #if procno == 0 and do_save_spectrum:
            # save a time sequence of velocity fields
        #    np.save(output_name + "energy", np.asarray(velocity_signal))
        if do_save_data and procno==0:
            data_chk.close()

        return np.asarray(kinetic_energy_series), np.asarray(potential_energy_series), np.asarray(total_energy_series), np.asarray(casimir_series), np.asarray(non_casimir_series)
                # index += 1
                # energy[index] = tqg_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
                # kinetic_energy[index] = tqg_kinetic_energy(q0, self.psi0, self.rotation)
                # potential_energy[index] = tqg_potential_energy(self.bathymetry, b0)
#                b_errors.append(errornorm(b0, self.initial_b, norm_type='L1'))
#                q_errors.append(errornorm(q0, self.initial_cond, norm_type='L1'))
#
#        q_err = errornorm(q0, self.initial_cond, norm_type = 'L1')
#
#        # np.save(output_name + "energy", energy)
#        # np.save(output_name + "kinetic_energy", kinetic_energy)
#        # np.save(output_name + "potential_energy", potential_energy)
#  
#        if (procno == 0):
#            np.save(output_name + "q_errors", np.asarray(q_errors))
#            np.save(output_name + "b_errors", np.asarray(b_errors))
#            try:
#                import matplotlib.pyplot as plt
#            except:
#                warning("Matplotlib not imported")
#
#            try:
#                fig, axes = plt.subplots()
#                axes.plot(b_errors, c='b', label='b error')
#                axes.plot(q_errors, c='r', label='q error')
#            except Exception as e:
#                warning("Cannot plot figure. Error msg: '%s'" % e)
#            try:
#                plt.legend()
#                plt.savefig(plot_output_name + ".png", dpi=100)
#                plt.close()
#            except Exception as e:
#                warning("Cannot show figure. Error msg: '%s'" % e)
