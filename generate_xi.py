from firedrake import *
from tqg.initial_conditions_abstract_class import TQGFemParams
import matplotlib.pyplot as plt
import numpy as np
from eofanalysis import eofsolver
from adjustText import adjust_text
from utility import Workspace

from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake_utility import TorusMeshHierarchy
# from firedrake import pi, ensemble,COMM_WORLD, PETSc
from firedrake.petsc import PETSc

from tqg.example2 import TQGExampleTwo as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm
global_comm = ensemble.global_comm

root = 0

def load_from_file(fdrake_function, filename, fieldname):
    with DumbCheckpoint(filename, mode=FILE_READ) as chk:
        chk.load(fdrake_function, name=fieldname)

def save_to_file(fdrake_function, filename):
    with DumbCheckpoint(filename, single_file=True, mode=FILE_CREATE) as chk:
        chk.store(fdrake_function)

def psi_file_name(file_index, workspace):
    return workspace.output_name("pde_data_{}".format(file_index), "PDESolution-no-bathymetry-perturbation")

def rpsi_file_name(file_index, workspace, increment_id=None):
    _fname_= "_regularised_psi_increment_{}".format(file_index)
    _fname = _fname_ + "_{}".format(increment_id) if increment_id !=None else _fname_ 
    return workspace.output_name(_fname, "CalibrationData128") 


class ComputeXi:

    def __init__(self):
        """
        :param _resolution: N, so spatial res is NxN
        :param _coarse_mesh: coarse mesh
        :param _dx_interval: interval to compute the trjaectory
        """
        self.dx_interval = 0 #_dx_interval
        self.nx = 512
        self.cnx = 128  # was 128
        self.alpha = 128 # was 100 
        #self.fname = "/home/wpan1/Data2/PythonProjects/TQGExample2/res_{}_alpha_{}".format(self.nx, self.alpha)
        #self.input_fname = "/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(self.nx, 'none')
        self.input_fname = "/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin"
        self.workspace = Workspace(self.input_fname)

    def compute_eof_data(self):
        """
        takes a time series of delta_x's (or delta_psi's), compute the eofs

        :return:
        """
        # calls svd solution
        # dx_ndays = utility.dx_ndays  # 0.02
        # data = np.zeros((ndays, 4225, 2))
        
        # ### load data #####
        nx = self.nx
        cnx = self.cnx
        T = 0.001
        dt = 0.0002
        alpha = self.alpha 
        dump_freq = 5 
        self.dx_interval = dt * dump_freq

        psi_name = "Streamfunction"
        rpsi_reg_name = "StreamfunctionRegularised"
        rpsi_name = "Streamfunction"

        mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
        params = Example2params(T,dt,mesh, bc='x', alpha=None)
        params_reg = Example2params(T,dt,mesh, bc='x', alpha=alpha)

        # fname = "/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha)
        # input_fname = "/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, 'none')
        # fname = self.fname
        input_fname = self.input_fname

        input_workspace = Workspace(input_fname)
        # workspace = Workspace(fname)
        workspace = self.workspace
        dpsi_firedrake_data_name = workspace.output_name("dpsi_firedrake_data", "CalibrationData128")       
        myprint = PETSc.Sys.Print

        solver = TQGSolver(params)
        solver_reg = RTQGSolver(params_reg) 
        psi_t_np = Function(solver.psi0.function_space(), name="Streamfunction")
        psi_t_n  = Function(solver.psi0.function_space())
        rpsi_reg_t_np = Function(solver_reg.psi0.function_space(), name="StreamfunctionReg")
        rpsi_reg_t_n  = Function(solver_reg.psi0.function_space())
        dpsi_reg_error = Function(solver.psi0.function_space(), name="StreamfunctionIncrementError")

        func = None
        coords_shape = None
        coords = None

        _inc = 20
        ndays = len(range(0, 1500, _inc)) 
        # data = np.zeros(((ndays,)+ dpsi_reg_error.dat.data.shape))
        # print(dpsi_reg_error.dat.data.shape)

        if spatial_comm.rank == root:
            mesh2 = TorusMeshHierarchy(cnx, cnx, 1., 1., 0, period="y", comm=ensemble_comm).get_fine_mesh()
            vfs = VectorFunctionSpace(mesh2, "CG", 1)
            fs = FunctionSpace(mesh2, "CG", 1)
            coords = Function(vfs).interpolate(SpatialCoordinate(vfs))
            func = Function(fs)
            coords_shape = coords.dat.data.shape

        coords_shape = spatial_comm.bcast(coords_shape, root=root)
        coords_all = np.zeros(coords_shape)
        if spatial_comm.rank == root:
            coords_all += coords.dat.data

        spatial_comm.Bcast(coords_all, root=root)
        data = np.zeros((ndays, len(coords_all)))
        #print(spatial_comm.rank , np.linalg.norm(coords_all), coords_all.shape, len(coords_all))
        #print(spatial_comm.rank, data.shape)


        for f_index, d_index in zip(range(0, 1500, _inc), range(ndays)):  # in increments of 5
            psi_filename_t_n = psi_file_name(f_index, input_workspace)
            psi_filename_t_np = psi_file_name(f_index+1, input_workspace)

            rpsi_filename = rpsi_file_name(f_index, workspace)
            rpsi_filename_t_n = rpsi_file_name(f_index, workspace, 0)
            rpsi_filename_t_np = rpsi_file_name(f_index, workspace, 1)

            myprint(psi_filename_t_n, coords_all.shape)

            load_from_file(psi_t_np, psi_filename_t_np , psi_name)
            load_from_file(psi_t_n , psi_filename_t_n , psi_name)
            load_from_file(rpsi_reg_t_np, rpsi_filename_t_np, rpsi_reg_name)
            load_from_file(rpsi_reg_t_n,  rpsi_filename_t_n , rpsi_reg_name)

            dpsi_reg_error.assign(0.5*dt*dump_freq*((psi_t_np + psi_t_n) - (rpsi_reg_t_n + rpsi_reg_t_np)))

            #print(norm(dpsi_reg_error))

            # data[d_index, :] += dpsi_reg_error.dat.data[:]
            data[d_index, :] += np.asarray(dpsi_reg_error.at(coords_all, tolerance=1e-10))

            #myprint(data[d_index,:])

        if spatial_comm.rank == root:
            #print("save", flush=True)
            np.save(dpsi_firedrake_data_name, data)
            myprint("save", data.shape)

    def compute_eof(self, scale=1): 
        data = np.load(self.workspace.output_name("dpsi_firedrake_data.npy", "CalibrationData128"))
        print(data.shape)
        cnx = self.cnx
        if 1:
            variances = [0.5, 0.7, 0.9]

            eof_solver = eofsolver.EofSolver(data)
            eofs = eof_solver.eofs(2) * scale   # scaled by sqrt(lambda)
            lambdas = eof_solver.eigenvalues()
            n_lambs = lambdas / np.sum(lambdas)

            nums = self.show_number_of_eofs(n_lambs, variances)
            self.show_data(lambdas, nums, 'spectrum', self.workspace.output_name('svd_spec_plot',"CalibrationData128"))
            self.show_data(n_lambs, nums, 'normalised spectrum', self.workspace.output_name('svd_norm_spec_plot',"CalibrationData128"))

        mesh2 = TorusMeshHierarchy(cnx, cnx, 1., 1., 0, period="y", comm=ensemble_comm).get_fine_mesh()
        vfs = VectorFunctionSpace(mesh2, "DG", 1)
        fs = FunctionSpace(mesh2, "CG", 1)

        gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))
        zeta35 = Function(fs, name="zeta35")
        xi35   = Function(vfs, name="xi35")
        zeta35.dat.data[:] = eofs[35]
        xi35.project(gradperp(zeta35))
        output_file = File(self.workspace.output_name("zeta_and_xi_35.pvd", "CalibrationData128"))
        output_file.write(zeta35, xi35)

        zeta0 = Function(fs, name="zeta0")
        xi0   = Function(vfs, name="xi0")
        zeta0.dat.data[:] = eofs[0]
        xi0.project(gradperp(zeta0))
        print( zeta0.dat.data.shape, eofs[0].shape, xi0.dat.data.shape)
 
        zeta1 = Function(fs, name="zeta1")
        xi1   = Function(vfs, name="xi1")
        zeta1.dat.data[:] = eofs[1]
        xi1.project(gradperp(zeta1))
 
        zeta2 = Function(fs, name="zeta2")
        xi2   = Function(vfs, name="xi2")
        zeta2.dat.data[:] = eofs[2]
        xi2.project(gradperp(zeta2))
 
        output_file = File(self.workspace.output_name("zetas_and_xis.pvd", "CalibrationData128"))
        output_file.write(zeta0, xi0, zeta1,xi1, zeta2,xi2)


        print(eofs.shape)
        np.save(self.workspace.output_name("zetas", "CalibrationData128"), eofs)
        
        # for num, var in zip(nums, variances):
        #     # print(num, var)
        #     _eof = eofs[:num]
        #     np.savetxt("{}/eof_{}.csv".format(self.zetas_dir, var), _eof.reshape((num, eofs.shape[1]*eofs.shape[2])), delimiter=",")
        #     self.zetasolver(eofs, num, var, self.dx_interval)
        #     np.savetxt("{}/zetas_{}.csv".format(self.zetas_dir, var), zetas[:_numeofs], delimiter=",")

    def show_data(self, data, nums, label, outputname='normalised_spec_plot'):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        assert len(nums) == 3

        colors = ['c', 'm', 'r']
        labels = [r'$n_{\bf{\xi}} \equiv 50\%$',
                  r'$n_{\bf{\xi}} \equiv 70\%$',
                  r'$n_{\bf{\xi}} \equiv 90\%$']

        ax.plot(data)
        for i in range(len(nums)):
            ax.scatter(nums[i], data[nums[i]], marker='o', color=colors[i], label=labels[i])

        # annotation
        annotations = []
        # for xy in zip(nums[2:], data[nums[2:]]):
        #     ax.annotate('%s' % xy[0], xy=xy, textcoords='data')

        for i, txt in enumerate(nums):
            annotations.append(ax.text(nums[i], data[nums[i]], txt))
        adjust_text(annotations)

        plt.ylabel(label, fontsize=12)
        plt.xlabel('number of EOFs', fontsize=12)
        plt.xlim([-10, 250])
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig("{}.svg".format(outputname), format='svg', dpi=1200)

    def show_number_of_eofs(self, normalised_eig_vals, target_percentages):
        result = []
        for target_percentage in target_percentages:
            assert target_percentage < 1.000000001
            i = 0
            percent = 0
            while percent < target_percentage:
                percent += normalised_eig_vals[i]
                i += 1
            result.append(i)
        return result


if __name__=="__main__":

    compxi = ComputeXi()
    #print("start")
    compxi.compute_eof_data() # may need to run this using mpi  because the data saved could be mpi
    # compxi.compute_eof()
    compxi.compute_eof(np.sqrt(0.001)) # T = 0.001, the xi's should have dimension length/sqrt(t)

