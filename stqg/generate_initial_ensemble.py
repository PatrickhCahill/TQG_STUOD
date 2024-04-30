#
#   Author: Wei Pan
#   Copyright   2021
#
#   Generate initial ensemble for STQG using random temporal rescaling
#   
#

from firedrake import *
from firedrake.petsc import PETSc
from utility import Workspace
from firedrake_utility import TorusMeshHierarchy
from tqg.solver import TQGSolver
from stqg.solver import STQGSolver
import numpy as np
from random import randint, sample


def pde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("pde_data_{}".format(file_index), sub_dir) 

def cpde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("cpde_data_{}".format(file_index), sub_dir)


###########################################################################################################################
#########
###########################################################################################################################
class STQGEnsembleGeneratorCoarse(STQGSolver):

    def __init__(self, tqg_params, truth_params, ensemble_size, comm_manager, batch_id=0, truth_ref_file_id=0, switch_off_bathymetry=False):
    #""" We assume tqg_params is defined using cPDE solver settings """

        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm  = comm_manager.comm

        #print(ensemble_comm.size, spatial_comm.size)

        STQGSolver.__init__(self, tqg_params)

        if switch_off_bathymetry == True:
            self.bathymetry.assign(0)

        self.ensemble_size = ensemble_size
        
        self.output_sub_dir = 'EnsembleMembers'
        self.output_name = "ensemble_member_{}".format(ensemble_comm.rank + batch_id*ensemble_comm.size)
        #print(self.output_name, flush=True)

        #self.epsilon = 1

        #self.file_index_range_for_perturbation = range(2000, 2775)  # file indices for perturbation
        self.file_work_space = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

        if ensemble_comm.rank == 0:
            print("load initial conditions", flush=True)

        # either we use the SPDE to generate the ensemble
        # or we use random temporal rescaling at the coarse level
        # in case 1), we load and corase grain the truth, then run an ensemble of spdes
        #   using the coarse level spde solver, without additional considerations
        # in case 2), we load the truth, coarse grain and project down,
        #   run the coarse version of pde solver, though in each step, we corase grain and project a randomly selected high res 
        
        self.load_initial_conditions_from_file(cpde_data_fname(self.file_work_space, truth_ref_file_id, "cPDESolution"), spatial_comm) 

        if ensemble_comm.rank == 0:
            print("STQGEnsembleGeneratorCoarse initialiser", flush=True)


    def generate_an_ensemble_member(self, comm_manager, dumpfreq=1, xi_scaling=1):
        visual_output_name = self.file_work_space.output_name('ensemble_member_{}'.format(comm_manager.ensemble_comm.rank), self.output_sub_dir) #"Ensembles")# visual output name
        data_output_name = ''
        zeta_file_name = self.file_work_space.output_name("zetas.npy", "ParamFiles")

        #print(comm_manager.ensemble_comm.rank, " solve", flush=True)

        self.solve(dumpfreq, visual_output_name, data_output_name, comm_manager, do_save_data=False, do_save_visual=False, do_save_spectrum=False, zetas_file_name = zeta_file_name, xi_scaling=xi_scaling) 
        # may need to rename ensembles
        data_output_name = self.file_work_space.output_name(self.output_name,self.output_sub_dir)

        #print("DumbCheckpoint")

        with DumbCheckpoint(data_output_name, single_file=True, mode=FILE_CREATE, comm=comm_manager.comm) as data_chk:
            data_chk.store(self.initial_cond)
            data_chk.store(self.initial_b)
            data_chk.store(self.psi0)
            data_chk.store(self.ssh)



###########################################################################################################################
#########
###########################################################################################################################

class STQGEnsembleGeneratorCoarseDan(STQGSolver):

    def __init__(self, tqg_params, ensemble_size, comm_manager, batch_id=0, truth_ref_file_id=0, switch_off_bathymetry=False):
    #""" We assume tqg_params is defined using cPDE solver settings """

        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm  = comm_manager.comm

        #print(ensemble_comm.size, spatial_comm.size)

        STQGSolver.__init__(self, tqg_params)

        if switch_off_bathymetry == True:
            self.bathymetry.assign(0)

        self.ensemble_size = ensemble_size
        
        self.output_sub_dir = 'EnsembleMembers'
        self.output_name = "ensemble_member_{}".format(ensemble_comm.rank + batch_id*ensemble_comm.size)
        #print(self.output_name, flush=True)

        #self.epsilon = 1

        #self.file_index_range_for_perturbation = range(2000, 2775)  # file indices for perturbation
        self.file_work_space = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

        if ensemble_comm.rank == 0:
            print("load initial conditions", flush=True)

        # either we use the SPDE to generate the ensemble
        # or we use random temporal rescaling at the coarse level
        # in case 1), we load and corase grain the truth, then run an ensemble of spdes
        #   using the coarse level spde solver, without additional considerations
        # in case 2), we load the truth, coarse grain and project down,
        #   run the coarse version of pde solver, though in each step, we corase grain and project a randomly selected high res 
        
        self.load_initial_conditions_from_file(cpde_data_fname(self.file_work_space, truth_ref_file_id, "cPDESolution"), spatial_comm) 
        # print("loaded")

        if ensemble_comm.rank == 0:
            print("STQGEnsembleGeneratorCoarseDan initialiser", flush=True)





###########################################################################################################################
#########
###########################################################################################################################
class STQGEnsembleGenerator(TQGSolver):

    """ We assume tqg_params is defined using PDE solver settings """
    def __init__(self, tqg_params, ensemble_size, comm_manager, batch_id=0, truth_ref_file_id=0, switch_off_bathymetry=False):
        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm  = comm_manager.comm

        #print(ensemble_comm.size, spatial_comm.size)

        TQGSolver.__init__(self, tqg_params)

        #switch off bathymetry forcing when generating initial conditions
        if switch_off_bathymetry == True:
            self.bathymetry.assign(0)

        self.ensemble_size = ensemble_size
        
        self.output_sub_dir = 'EnsembleMembers'
        self.output_name = "ensemble_member_{}".format(ensemble_comm.rank + batch_id*ensemble_comm.size)
        #print(self.output_name, flush=True)

        self.epsilon = 1

        self.file_index_range_for_perturbation = range(2775) # range(2000, 2775)  # file indices for perturbation

        self.file_work_space = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

        if ensemble_comm.rank == 0:
            print("load initial conditions", flush=True)

        self.load_initial_conditions_from_file(pde_data_fname(self.file_work_space, truth_ref_file_id, "PDESolution"), spatial_comm)

        #print("loaded")

        if ensemble_comm.rank == 0:
            print("STQGEnsembleGenerator initialiser", flush=True)
        

    def _sample_and_scatter_data_file_index(self, comm):
        procno_ecomm = comm.rank
        root=0
        sendbuf_findex = None
        sendbuf_randnum = None
        if procno_ecomm == root:
            #sendbuf_findex = [randint(self.file_index_range_for_perturbation[0]
            #    , self.file_index_range_for_perturbation[-1]+1) for i in range(self.ensemble_size)]
            #sendbuf_findex = np.random.randint(self.file_index_range_for_perturbation[0]
            #        , self.file_index_range_for_perturbation[-1]+1
            #        , size=self.ensemble_size)

            sendbuf_findex = np.asarray(sample(self.file_index_range_for_perturbation, self.ensemble_size))
            #sendbuf_findex = np.ones(self.ensemble_size, dtype=np.int) 

            sendbuf_randnum = np.random.randn(self.ensemble_size) * self.epsilon
            
            #print(sendbuf_findex, sendbuf_randnum)

            #print(sendbuf_randnum)
        recvbuf = np.empty(1, dtype=np.uint)
        #recvbuf = np.empty(1)
        recvbuf_randnum = np.empty(1, dtype=np.double)

        #ensemble_comm.Scatter(sendbuf_findex, recvbuf, root=root)
        comm.Scatter(sendbuf_findex, recvbuf, root=root)
        comm.Scatter(sendbuf_randnum, recvbuf_randnum, root=root)
        #recvbuf = ensemble_comm.scatter(sendbuf_findex, root=root)

        #print("ensemble procno: ", procno_ecomm, ", ", recvbuf)
        #print("ensemble procno: ", procno_ecomm, ", ", recvbuf_randnum)
        
        return recvbuf[0], recvbuf_randnum[0], sendbuf_findex



    def generate_an_ensemble_member(self, cnx, cny, comm_manager, dumpfreq=1):
        """
        generate an initial ensemble using random temporal rescaling

        work on fine resolution scale

        saves data in 

        :return: 
        """
        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm  = comm_manager.comm

        PETSc.Sys.Print('STQG ensemble generator',flush=True)
        
        # assume q0 is given in initial_cond
        # high res data use spatial parallelism !!!!!
        q0 = Function(self.Vdg, name="PotentialVorticity")
        b0 = Function(self.Vdg, name="Buoyancy")
        self.psi0.rename("Streamfunction")
        q0.assign(self.initial_cond)
        b0.assign(self.initial_b)
        Dt = self.Dt
        root = 0

        #coords_shape = spatial_comm.bcast(coords_shape, root=root)
        #coords_all = np.zeros(coords_shape)
        #if spatial_comm.rank == root:
        #    coords_all += coords.dat.data

        #spatial_comm.Bcast(coords_all, root=root)
        #data = np.zeros((ndays, len(coords_all)))

        #self.q1.assign(q0)
        #self.psi_solver.solve()
        #v.project(self.gradperp(self.psi0))
        
        # ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
        procno_ecomm = ensemble_comm.rank
        procno_scomm = spatial_comm.rank
        
        #output_file = None
        #data_chk = None
        
        #compute_kinetic_energy = lambda _w, _psi, _f : assemble( -0.5 * _psi * (_w - _f ) *dx )
        #compute_potential_energy = lambda _h, _b : assemble( -0.5* _h * _b * dx)
        #compute_total_energy = lambda _w, _psi, _f, _h, _b : assemble( -0.5 * (_psi*(_w-_f) + _h*_b ) *dx ) 
        #compute_casimir = lambda _w, _b : assemble( (_b + _w*_b)*dx )
        #compute_non_casimir = lambda _w, _b : assemble( (_b + _w*_w*_b) *dx )

        #kinetic_energy_series = []
        #potential_energy_series = []
        #total_energy_series = []
        #casimir_series = []
        #non_casimir_series = []
        
        #if (procno_ecomm == 0):
            #_ke = compute_kinetic_energy(q0, self.psi0, self.rotation)
            #_pe = compute_potential_energy(self.bathymetry, b0)
            #_te = compute_total_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
            #kinetic_energy_series.append(_ke)
            #potential_energy_series.append(_pe)
            #total_energy_series.append(_te)
            #casimir_series.append(compute_casimir(q0, b0))
            #non_casimir_series.append(compute_non_casimir(q0, b0))
            
            # print(_te, _ke, _pe, flush=True)
        
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

        #mesh_grid = tqg_mesh_grid(res)
        #index = 0

        # start_time = time.time()
        # b_errors = []
        # q_errors = []
        #if 1:
        #print(procno_ecomm, T, Dt, flush=True)
        _findex, beta, sentindices = self._sample_and_scatter_data_file_index(ensemble_comm)
        #h5_data_name = pde_data_fname(self.file_work_space, _findex, 'PDESolution/Perturbation')
        #with DumbCheckpoint(h5_data_name, mode=FILE_READ, comm=spatial_comm) as chk:
        #    chk.load(self.psi0, name="Streamfunction")

        while t < (T - Dt / 2) and 1:
            self.b1.assign(b0)
            self.q1.assign(q0)
            self.db2.project(b0)

            # load in psi from file
            _findex, _beta, sentindices = self._sample_and_scatter_data_file_index(ensemble_comm)
            h5_data_name = pde_data_fname(self.file_work_space, _findex, 'PDESolution/Perturbation')

            if procno_ecomm == 0 and 0:
                print(procno_ecomm, h5_data_name, flush=True)

            with DumbCheckpoint(h5_data_name, mode=FILE_READ, comm=spatial_comm) as chk:
                chk.load(self.psi0, name="Streamfunction")

            #print(procno_ecomm, "before ", norm(self.psi0), beta, flush=True)
            self.psi0.assign(self.psi0*beta)
            #print(procno_ecomm, "after ", norm(self.psi0), flush=True)

            # self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()     

            # # Find intermediate solution q^(1)
            self.b1.assign(self.db1)
            self.q1.assign(self.dq1)
            self.db2.project(self.b1)
            
            #self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()

            # # Find intermediate solution q^(2)
            self.b1.assign(0.75 * b0 + 0.25 * self.db1)
            self.q1.assign(0.75 * q0 + 0.25 * self.dq1)
            self.db2.project(self.b1)

            #self.psi_solver.solve()
            self.b_solver.solve()
            self.q_solver.solve()

            # # Find new solution q^(n+1)
            b0.assign(b0 / 3 + 2 * self.db1 / 3)
            q0.assign(q0 / 3 + 2 * self.dq1 / 3)

            # Store solutions to xml and pvd
            t += Dt
            tdump += 1
            _t = round(t, 5)
            #PETSc.Sys.Print(_t,"/",T, flush=True)
            if tdump == dumpfreq:
                tdump -= dumpfreq
                if procno_ecomm == 0:
                    #print("end of loop", flush=True)
                    print(_t, flush=True)
            #    _ke, _pe, _te = 0, 0, 0 
            #    if (procno_ecomm == 0):
            #        self.q1.assign(q0)
            #        self.psi_solver.solve()
            #        v.project(self.gradperp(self.psi0))

            #        _ke = compute_kinetic_energy(q0, self.psi0, self.rotation)
            #        _pe = compute_potential_energy(self.bathymetry, b0)
            #        _te = compute_total_energy(q0, self.psi0, self.rotation, self.bathymetry, b0)
            #        kinetic_energy_series.append(_ke)
            #        potential_energy_series.append(_pe)
            #        total_energy_series.append(_te)
            #        non_casimir_series.append(compute_non_casimir(q0, b0))

            #        if do_save_visual:
            #            ssh.assign(Function(self.Vdg).project(self.psi0) - 0.5 * b0)
            #            output_file.write(q0, self.psi0, v, b0, ssh, time=_t)
            #        if do_save_spectrum:                        
            #            # -------- FFT --------------------------------------
            #            index += 1
            #            np.save(data_output_name + "_energy_{}".format(index), np.asarray(v.at(mesh_grid, tolerance=1e-10)))                            
            #           # -------- FFT end --------------------------------------
            #        if do_save_data:
            #            data_chk.new_file()
            #            data_chk.store(q0)
            #            data_chk.store(b0)
            #            data_chk.store(self.psi0)  # for calibration
                #PETSc.Sys.Print(_t,"/",T, " ", _te, _ke, _pe, flush=True)
        # end loop

        #if procno_ecomm == 0 and do_save_spectrum:
            # save a time sequence of velocity fields
        #    np.save(output_name + "energy", np.asarray(velocity_signal))
        #if do_save_data and procno_ecomm==0:
        #    data_chk.close()

        #if do_save_visual:
        #    output_file = File(output_name + ".pvd")

        self.q1.assign(q0)
        self.psi_solver.solve()

        ## project down to coarse grid and save data 
        if 0:
            print("e procno: ", procno_ecomm, ", done solving.")
            fdata_output_name = self.file_work_space.output_name(self.output_name + "_fs", self.output_sub_dir)
            foutput_file = File(fdata_output_name + ".pvd", comm=spatial_comm)
            foutput_file.write(q0, self.psi0, b0)
            with DumbCheckpoint(fdata_output_name, single_file=True, mode=FILE_CREATE, comm=spatial_comm) as fchk:
                fchk.store(q0)
                fchk.store(b0)
                fchk.store(self.psi0)
        if 1:
            cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
            cq0 = self.coarse_grain_and_project(q0, cnx, cny, cmesh, spatial_comm)
            cb0 = self.coarse_grain_and_project(b0, cnx, cny, cmesh, spatial_comm)
            cpsi0 = self.coarse_grain_and_project(self.psi0, cnx, cny, cmesh, spatial_comm)

            cssh = Function(FunctionSpace(cmesh, "CG", 1), name="SSH").assign(cpsi0 - 0.5*cb0)

            cfs_dg = FunctionSpace(cmesh, "DG", 1)
            cq0 = project(cq0, cfs_dg)
            cb0 = project(cb0, cfs_dg)

            cq0.rename("PotentialVorticity")
            cb0.rename("Buoyancy")
            cpsi0.rename("Streamfunction")

            data_output_name = self.file_work_space.output_name(self.output_name,self.output_sub_dir)
            output_file = File(data_output_name + ".pvd", comm=spatial_comm)
            output_file.write(cq0, cpsi0, cb0)

            with DumbCheckpoint(data_output_name, single_file=True, mode=FILE_CREATE, comm=spatial_comm) as data_chk:
                data_chk.store(cq0)
                data_chk.store(cb0)
                data_chk.store(cpsi0)
                data_chk.store(cssh)

            #data_chk.close()

        #return np.asarray(kinetic_energy_series), np.asarray(potential_energy_series), np.asarray(total_energy_series), np.asarray(casimir_series), np.asarray(non_casimir_series)


    def coarse_grain_and_project(self, func_fine_res, cnx, cny, cmesh, spatial_comm, field="Streamfunction"):
        """
        coarse grained data are CG functions
        """

        k_sqr =cnx * cny  #cnx * cny * 256 / cnx * 256 / cny 

        flag_direct_subsample = False

        #Helmhotlz solver -- u is the solution
        u = None
        fs = func_fine_res.function_space()
        cg_fs = FunctionSpace(fs.mesh(), 'CG', 1)
        f = Function(cg_fs).project(func_fine_res)

        if flag_direct_subsample == False:
            u = TrialFunction(cg_fs)
            v = TestFunction(cg_fs)
            c_sqr = Constant(1.0 / k_sqr)

            a = (c_sqr * dot(grad(v), grad(u)) + v * u) * dx
            L = f * v * dx

            bc = DirichletBC(cg_fs, f, 'on_boundary') #if field == "Streamfunction" else []

            u = Function(cg_fs)
            solve(a == L, u, bcs=bc, solver_parameters={'ksp_type': 'cg'})
        else:
            u = f

        #project
        #cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
        cvfs = VectorFunctionSpace(cmesh, "CG", 1)
        cfs = FunctionSpace(cmesh, "CG", 1)
        ccoords = Function(cvfs).interpolate(SpatialCoordinate(cvfs))
        cfunc = Function(cfs)
        u_at_values = np.asarray(u.at(ccoords.dat.data, tolerance=1e-10))
        assert cfunc.dat.data.shape == u_at_values.shape 

        cfunc.dat.data[:] += u_at_values 

        return cfunc


    def helmholtz_solver(func, k_sqr=64.0):
        """ 
        assumes alpha = 1   (I - c. \Laplace)func_bar = func
        solving for func_bar which is like an averaged version of func
        
        this averages out frequencies which are less than c.
        """
        fs = func.function_space()
        cg_fs = FunctionSpace(fs.mesh(), 'CG', 1)
        f = Function(cg_fs).project(func)

        u = TrialFunction(cg_fs)
        v = TestFunction(cg_fs)
        c_sqr = Constant(1.0 / k_sqr)

        a = (c_sqr * dot(grad(v), grad(u)) + v * u) * dx
        L = f * v * dx

        bc = DirichletBC(cg_fs, 0.0, 'on_boundary')

        u = Function(cg_fs)
        solve(a == L, u, bcs=bc, solver_parameters={'ksp_type': 'cg'})

        return func.project(u) 


        msh = psi.function_space().mesh()
        q_cg_space = FunctionSpace(msh, 'CG', 1)

        q = TrialFunction(q_cg_space)
        v = TestFunction(q_cg_space)

        bc = DirichletBC(q_cg_space, 0.0, 'on_boundary')

        a = -(dot(grad(v), grad(psi))) * dx
        L = q * v * dx

        q0 = Function(q_cg_space)
        solve(L == a, q0, bcs=[bc])
        q_dg.project(q0)
        

    def generate_an_ensemble_member_Dan(self, cnx, cny, comm_manager, dumpfreq=1, epsilon=1.):
        #print("epsilon: ", epsilon, flush=True)
        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm = comm_manager.comm
        
        q0 = Function(self.Vdg, name="PotentialVorticity")
        b0 = Function(self.Vdg, name="Buoyancy")
        self.psi0.rename("Streamfunction")
        q0.assign(self.initial_cond)
        b0.assign(self.initial_b)

        if 1:
            cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
            cq0 = self.coarse_grain_and_project(q0, cnx, cny, cmesh, spatial_comm)
            cb0 = self.coarse_grain_and_project(b0, cnx, cny, cmesh, spatial_comm)
            cpsi0 = self.coarse_grain_and_project(self.psi0, cnx, cny, cmesh, spatial_comm)

            # 1scaling = np.ones(cq0.dat.data.shape) 
            gscaling = np.random.normal(1., epsilon, cq0.dat.data.shape) 
            #print("gscaling[0]: ", gscaling[0],flush=True)

            #print(cq0.dat.data.shape, gscaling.shape, flush=True)
            
            assert (gscaling.shape == cq0.dat.data.shape), "cq0 shape != gscaling"
            assert (gscaling.shape == cb0.dat.data.shape), "cb0 shape != gscaling"
            assert (gscaling.shape == cpsi0.dat.data.shape), "cpsi0.shape != gscaling"

            #print(type(cq0.dat.data), gscaling, flush=True)

            cq0.dat.data[:] *= gscaling
            cb0.dat.data[:] *= gscaling
            cpsi0.dat.data[:] *= gscaling
            cssh = Function(FunctionSpace(cmesh, "CG", 1), name="SSH").assign(cpsi0 - 0.5*cb0)

            cfs_dg = FunctionSpace(cmesh, "DG", 1)
            cq0 = project(cq0, cfs_dg)
            cb0 = project(cb0, cfs_dg)

            cq0.rename("PotentialVorticity")
            cb0.rename("Buoyancy")
            cpsi0.rename("Streamfunction")

            data_output_name = self.file_work_space.output_name(self.output_name,self.output_sub_dir)
            output_file = File(data_output_name + ".pvd", comm=spatial_comm)
            output_file.write(cq0, cpsi0, cb0, cssh)

            with DumbCheckpoint(data_output_name, single_file=True, mode=FILE_CREATE, comm=spatial_comm) as data_chk:
                data_chk.store(cq0)
                data_chk.store(cb0)
                data_chk.store(cpsi0)
                data_chk.store(cssh)

