#
# Author Wei Pan
# Edited by Patrick Cahill
# STQG solver
#


from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
from abc import ABC, abstractmethod
import os,sys, shutil
from tqdm import tqdm
import warnings


def tqg_energy(q, psi, f, h, b):
    return 0.5*assemble( (psi*(q-f) + h*b)*dx )

def tqg_kinetic_energy(q, psi, f):
    return 0.5 * assemble( psi*(q-f)*dx )

def tqg_potential_energy(h, b):
    return 0.5 * assemble( h*b*dx )

def tqg_mesh_grid(res):
    xx = np.linspace(0, 1, res+1)
    yy = np.linspace(0, 1, res+1)

    mygrid = []
    for i in range(len(xx)):
        for j in range(len(yy)):
            mygrid.append([xx[i], yy[j]])
    return np.asarray(mygrid) # now i have a list of grid values

class SWParams():
    """ firedrake finite element parameters """

    def __init__(self, dt, mesh, t, bc='y', cg_deg=1, alpha=None):
        self.Vdg = FunctionSpace(mesh, "DG", 1) # For potential vorticity and buoyancy and height
        self.Vcg = FunctionSpace(mesh, "CG", cg_deg) # For streamfunction, bathymetry and ssh
        self.Vu = VectorFunctionSpace(mesh, "DG", 1) # For velocity
        self.x = SpatialCoordinate(mesh)
        self.dt = dt
        self.facet_normal = FacetNormal(mesh)
        self.mesh = mesh
        
        self.time_length = t
        if (bc == 'y') or ( bc == 'x'):
            self.bcs = DirichletBC(self.Vcg, 0.0, 'on_boundary')
            self.dgbcs = DirichletBC(self.Vdg, 0.0, 'on_boundary')
        else:
            self.bcs = []
            self.dgbcs =[]

        self.initial_q = Function(self.Vdg, name="PotentialVorticity")
        self.initial_b = Function(self.Vdg, name="Buoyancy")
        self.bathymetry = Function(self.Vcg, name="Bathymetry")
        self.rotation = Function(self.Vdg, name="Rotation")
        self.alphasqr = Constant(1./alpha**2) if alpha != None else Constant(0)
        super().__init__()

    def set_initial_conditions(self):
        x= SpatialCoordinate(self.Vdg.mesh())
        self.initial_q.interpolate( -exp(-5.0*(2*pi*x[1] - pi)**2) )
        self.bathymetry.interpolate( cos(2*pi*x[0]) + 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*x[0])/3.0)
        self.rotation.assign(0)
        self.initial_b.interpolate(sin(2*pi*x[0]))

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation

    def set_forcing(self):
        return Function(self.Vdg).assign(0) 

    def set_damping_rate(self):
        return Constant(0.)

class STQGSolver():

    def __init__(self, sw_params, bathymetry_xi_scaling=1.):
        self.id = 0

        #TQGSolver.__init__(self, tqg_params)
        self.gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))
        
        self.time_length = sw_params.time_length

        #self.bathymetry is a CG function
        self.initial_cond, self.initial_b, self.bathymetry, self.rotation  = sw_params.set_initial_conditions()
        self.bathymetry_xi_scaling = bathymetry_xi_scaling

        self.Vdg = sw_params.Vdg
        self.Vcg = sw_params.Vcg
        self.Vu  = sw_params.Vu

        self.psi0 = Function(self.Vcg, name="Streamfunction")
        self.ssh  = Function(self.Vcg, name="SSH")

        self.dq1  = Function(self.Vdg)  
        self.q1   = Function(self.Vdg)
        
        self.db1 = Function(self.Vdg)  
        self.db2 = Function(self.Vcg)
        self.b1  = Function(self.Vdg)

        self.Dt = sw_params.dt
        self.mesh = sw_params.mesh
        dt      = Constant(self.Dt)

        psi = TrialFunction(self.Vcg)
        phi = TestFunction(self.Vcg)

        # --- elliptic equation ---
        Apsi = (dot(grad(phi), grad(psi)) + phi * psi) * dx
        Lpsi = (self.rotation - self.q1) * phi * dx 

        psi_problem = LinearVariationalProblem(Apsi, Lpsi, self.psi0, sw_params.bcs)
        self.psi_solver = LinearVariationalSolver(psi_problem, \
            solver_parameters={
                'ksp_type':'cg',
                'pc_type':'sor'
            }
        )

        # --- b equation -----
        un = 0.5 * (dot(self.gradperp(self.psi0), sw_params.facet_normal) \
            + abs(dot(self.gradperp(self.psi0), sw_params.facet_normal)))

        _un_ = dot(self.gradperp(self.psi0), sw_params.facet_normal)
        _abs_un_ = abs(_un_)

        b = TrialFunction(self.Vdg)
        p = TestFunction(self.Vdg)
        a_mass_b = p * b * dx
        a_int_b = (dot(grad(p), -self.gradperp(self.psi0) * b)) * dx

        a_flux_b =  0.5*jump(p)*(2*_un_('+')*avg(b) + _abs_un_('+')*jump(b))*dS  

        arhs_b = a_mass_b - dt * (a_int_b + a_flux_b)

        b_problem = LinearVariationalProblem(a_mass_b, action(arhs_b, self.b1), self.db1 \
                 , bcs=sw_params.dgbcs)  # solve for db1
        self.b_solver = LinearVariationalSolver(b_problem, \
            solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
            }
        )

        # --- q equation -----
        q = TrialFunction(self.Vdg)
        p_ = TestFunction(self.Vdg)
        a_mass_ = p_ * q * dx

        a_int_ = ( dot(grad(p_), -self.gradperp(self.psi0)*(q - self.db1))
                  + p_ * div(self.db2 * self.gradperp(0.5*self.bathymetry)) ) *dx

        a_flux_ =  0.5*jump(p_)*(2*_un_('+')*avg(q-self.db1) + _abs_un_('+')*jump(q - self.db1))*dS  

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
        self.x = sw_params.x
        self.Vcg = sw_params.Vcg
        self.solver_name = 'STRSW solver'

    def visualise_h5(self, h5_data_name_prefix,  output_visual_name, time_start=0, time_end=0, time_increment=0, initial_index=0):
        output_file = VTKFile(output_visual_name + ".pvd") 
        self.load_initial_conditions_from_file(f"{h5_data_name_prefix}_{initial_index}")

        Vu = VectorFunctionSpace(self.mesh, "CG",1)
        Vcg = FunctionSpace(self.mesh, "CG",1)
        v = Function(Vu, name="Velocity")
        v.assign(project(self.gradperp(self.psi0),Vu))
        self.ssh.assign(self.psi0 - 0.5 * project(self.initial_b, Vcg))

        output_file.write(self.initial_cond, self.psi0, v, self.initial_b, self.ssh, time=time_start)

        for t in np.arange(time_increment, time_end, time_increment):
            initial_index += 1
            print(f"{h5_data_name_prefix}_{initial_index}", flush=True)
            self.load_initial_conditions_from_file(f"{h5_data_name_prefix}_{initial_index}")
            Vu = VectorFunctionSpace(self.mesh, "CG",1)
            Vcg = FunctionSpace(self.mesh, "CG",1)
            v = Function(Vu, name="Velocity")
            v.assign(project(self.gradperp(self.psi0),Vu))
            self.ssh.assign(self.psi0 - 0.5 * project(self.initial_b, Vcg))
            output_file.write(self.initial_cond, self.psi0, v, self.initial_b, self.ssh, time=t)

    def animate(self, output_visual_name, save_name, scalar=None):
        import pyvista as pv
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        filepaths = []
        for i in sorted(os.listdir(output_visual_name)):
            if ".vtu" in i:
                filepaths.append(f"{output_visual_name}/{i}")
        pv.OFF_SCREEN = True


        images = []  # will store arrays for Matplotlib

        for fp in filepaths:
            mesh = pv.read(fp)

            plotter = pv.Plotter(off_screen=True)
            if scalar is not None:
                plotter.add_mesh(mesh, scalars=scalar, cmap="viridis")
            else:
                plotter.add_mesh(mesh, cmap="viridis")

            plotter.camera_position = "iso"
            
            # Returns an RGBA numpy array
            img = plotter.screenshot(return_img=True)
            images.append(img)

            plotter.close()


        fig, ax = plt.subplots()
        im = ax.imshow(images[0])
        ax.axis("off")

        def update(i):
            im.set_array(images[i])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(images), interval=200)

        anim.save(f"{output_visual_name}/{save_name}.mp4", fps=5, dpi=150)

    def load_initial_conditions_from_file(self, h5_data_name):
        """
        Read in initial conditions from saved (checkpointed) data file

        Loads pv, buoyancy and streamfunction
        """
        # PETSc.Sys.Print(norm(self.initial_cond))
        # with DumbCheckpoint(h5_data_name, mode=FILE_READ) as chk:
        #     chk.load(self.initial_cond, name="PotentialVorticity")
        #     chk.load(self.initial_b, name="Buoyancy")
        #     chk.load(self.psi0, name="Streamfunction")
        #     try:
        #         chk.load(self.ssh, name="SSH")
        #     except:
        #         print('no ssh in data')
        #         #pass

        with CheckpointFile(f"{h5_data_name}.h5", "r") as chk:
            self.mesh = chk.load_mesh("mesh")
            self.initial_cond = chk.load_function(self.mesh, name="PotentialVorticity")
            self.initial_b = chk.load_function(self.mesh, name="Buoyancy")
            self.psi0 = chk.load_function(self.mesh, name="Streamfunction")
            try:
                self.ssh = chk.load_function(self.mesh, name="SSH")
            except:
                print('no ssh in data')
                #pass

        # PETSc.Sys.Print(norm(self.initial_cond))

    def save_velocity_grid_data(self, h5_data_name, res):
        """
        Takes checkpoint data (pv, buoyancy) and save corresponding velocity field grid values for spectral analysis
        h5_data_name must end in '.h5'
        """

        np.save(h5_data_name, self.get_velocity_grid_data(h5_data_name, res)) 

    def get_velocity_grid_data(self, h5_data_name, res):
        """
        Takes checkpoint data (pv, buoyancy) and save corresponding velocity field grid values for spectral analysis
        h5_data_name must end in '.h5'
        """
        mesh_grid = tqg_mesh_grid(res)
        q0 = Function(self.Vdg)
        # with DumbCheckpoint(h5_data_name, mode=FILE_READ) as chk:
        #     chk.load(q0, name="PotentialVorticity")

        with CheckpointFile(f"{h5_data_name}.h5", "r") as chk:
            mesh = chk.load_mesh("mesh")
            q0 = chk.load_function(mesh, name="PotentialVorticity")

        v = Function(VectorFunctionSpace(self.Vdg.mesh(),"CG", 1), name="Velocity")
        self.q1.assign(q0)
        self.psi_solver.solve()
        v.project(self.gradperp(self.psi0))
        # PETSc.Sys.Print(norm(v), v.at([0.5, 0.5], tolerance=1e-10), flush=True)
        # PETSc.Sys.Print(h5_data_name)

        # Use PointEvaluator instead of deprecated Function.at
        from firedrake import PointEvaluator
        pe = PointEvaluator(v.function_space().mesh(), mesh_grid, tolerance=1e-10)
        return pe.evaluate(v)

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
        # with DumbCheckpoint(h5_data_name, mode=FILE_READ) as chk:
        #     chk.load(q0, name="PotentialVorticity")

        with CheckpointFile(f"{h5_data_name}.h5", "r") as chk:
            mesh = chk.load_mesh("mesh")
            q0 = chk.load_function(mesh, name="PotentialVorticity")

        self.q1.assign(q0)
        self.psi_solver.solve()



        # Use PointEvaluator for single-point evaluation
        from firedrake import PointEvaluator
        pts = np.atleast_2d(grid_point)
        pe = PointEvaluator(self.psi0.function_space().mesh(), pts, tolerance=1e-10)
        vals = pe.evaluate(self.psi0)
        return np.asarray(vals)[0]

    def solve(self, dumpfreq, output_name, data_output_name, comm_manager, do_save_data=False, do_save_visual=True, do_save_spectrum=False, res=0, zetas_file_name=None, xi_scaling=1, bathymetry_xi=False, procno=0, **kwargs):
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
        PETSc.Sys.Print(self.solver_name,flush=True)
        
        # assume q0 is given in initial_cond
        q0 = Function(self.Vdg)
        b0 = Function(self.Vdg)
        q0.assign(self.initial_cond)
        b0.assign(self.initial_b)
        

        Vu = self.Vu
        Dt = self.Dt

        q0.rename("PotentialVorticity")
        self.psi0.rename("Streamfunction")
        b0.rename("Buoyancy")
        v = Function(Vu, name="Velocity")
        
        self.q1.assign(q0)
        self.psi_solver.solve()
        v.project(self.gradperp(self.psi0))
        self.ssh.assign(self.psi0 - 0.5 * Function(self.Vcg).project(b0))
        
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

        # Always compute initial energies if appropriate and always open outputs
        if self.solver_name == 'TQG solver':
            _ke = compute_kinetic_energy(q0, self.psi0, self.rotation)
            _pe = compute_potential_energy(self.bathymetry, b0)
            _te = compute_total_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
            kinetic_energy_series.append(_ke)
            potential_energy_series.append(_pe)
            total_energy_series.append(_te)
            casimir_series.append(compute_casimir(q0, b0))
            non_casimir_series.append(compute_non_casimir(q0, b0))

        # always create visual output and checkpoint (may raise if invalid paths)
        output_file = VTKFile(output_name + ".pvd")
        output_file.write(q0, self.psi0, v, b0, self.ssh, time=0)
    
        # store initial snapshot with explicit names
        index = 0
        spef_data_output_name = f"{data_output_name}_{index}.h5"
        with CheckpointFile(spef_data_output_name, mode="w") as data_chk_alt:
            data_chk_alt.save_mesh(self.mesh)
            data_chk_alt.save_function(q0, name="PotentialVorticity")
            data_chk_alt.save_function(b0, name="Buoyancy")
            data_chk_alt.save_function(self.psi0, name="Streamfunction")
            data_chk_alt.save_function(self.ssh, name="SSH")




        
        t = 0.
        T = self.time_length
        tdump = 0

        mesh_grid = tqg_mesh_grid(res)

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
        if self.solver_name == 'STQG solver':
            #print(zetas_file_name)
            #zetas = xi_scaling * np.load(zetas_file_name) if zetas_file_name is not None else  np.asarray([Function(self.Vcg).dat.data])

            if (zetas_file_name is not None) and (bathymetry_xi is False):
                #print('zeta is from file', flush=True)
                zetas = xi_scaling * np.load(zetas_file_name)
            else:
                if bathymetry_xi is True:
                    #print('zetas = bathymetry', flush=True)
                    zetas = xi_scaling * np.asarray([self.bathymetry.dat.data]) 
                else:
                    #print('no zeta', flush=True)
                    zetas = np.asarray([Function(self.Vcg).dat.data])
            #print(zetas.shape)
            #Function(self.Vcg)
            #zetas = np.asarray([zeta.dat.data])  

            if 'proposal_step' in kwargs and 'state_store' in kwargs:
                noise = rho * state_store + np.sqrt((1. - rho**2) /Dt) * np.random.normal(0, 1, zetas.shape[0] * iter_steps)
                
            else:
                noise = np.sqrt(Dt)**(-1) * np.random.normal(0., 1., zetas.shape[0] * iter_steps) if ((zetas_file_name is not None) or (bathymetry_xi is True)) else np.zeros(zetas.shape[0] *  iter_steps)
                #print(noise, flush=True)
            #print(noise.shape, np.asarray(self.psi0.dat.data).shape)

        step = 0
        # while t < (T - Dt / 2):
        for t in tqdm(np.arange(t,T,Dt)):
            # sort out BM
            if self.solver_name == 'STQG solver':
                bms = noise[step:step+zetas.shape[0]]
                step += zetas.shape[0]

            # Compute the streamfunction for the known value of q0
            self.b1.assign(b0)
            self.q1.assign(q0)
            self.db2.project(b0)

            self.psi_solver.solve()
            psi0_perturbation = 0 if zetas is None else np.sum((zetas.T * bms).T, axis=0) 
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
            # t += Dt
            tdump += 1
            if tdump == dumpfreq:
                tdump -= dumpfreq
                _t = round(t, 5)
                _ke, _pe, _te = 0, 0, 0 
                # always update solver state and save outputs
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

                # always write visualisation
                output_file.write(q0, self.psi0, v, b0, self.ssh, time=_t)
                # always save spectrum snapshot
                index += 1

                from firedrake import PointEvaluator
                pe = PointEvaluator(v.function_space().mesh(), mesh_grid, tolerance=1e-10)
                arr = pe.evaluate(v)


                np.save(f"{data_output_name}_energy_{index}", arr)

                spef_data_output_name = f"{data_output_name}_{index}.h5"
                with CheckpointFile(spef_data_output_name, mode="w") as data_chk_alt:
                    data_chk_alt.save_mesh(self.mesh)
                    data_chk_alt.save_function(q0, name="PotentialVorticity")
                    data_chk_alt.save_function(b0, name="Buoyancy")
                    data_chk_alt.save_function(self.psi0, name="Streamfunction")
                    data_chk_alt.save_function(self.ssh, name="SSH")

                #PETSc.Sys.Print(_t,"/",T, " ", _te, _ke, _pe, flush=True)
        self.initial_cond.assign(q0)
        self.initial_b.assign(b0)

        # close checkpoint if opened
        try:
            if data_chk is not None:
                data_chk.close()
        except Exception:
            pass

        return noise

if __name__ == "__main__":
    OMP_NUM_THREADS = 1
    if os.path.exists("./output"):
        shutil.rmtree("./output")
    mesh = UnitSquareMesh(32, 32, name="mesh")
    sw_params = SWParams(dt=0.01, mesh=mesh, t=1, bc='y', cg_deg=1, alpha=0.1)
    stqg_solver = STQGSolver(sw_params)
    stqg_solver.solve(dumpfreq=10, output_name="output/strsw_test", data_output_name="output/strsw_data_test", comm_manager=None, do_save_data=True, do_save_visual=True, do_save_spectrum=False, res=16)
    stqg_solver.visualise_h5(h5_data_name_prefix="output/strsw_data_test", output_visual_name="output/strsw_test_visual", time_start=0, time_end=1, time_increment=0.1, initial_index=0)
    stqg_solver.animate("output/strsw_test_visual", "vorticity_anim", "PotentialVorticity")
    stqg_solver.animate("output/strsw_test_visual", "ssh_anim", "SSH")
    stqg_solver.animate("output/strsw_test_visual", "buoyancy_anim", "Buoyancy")
    stqg_solver.animate("output/strsw_test_visual", "streamfunction_anim", "Streamfunction")