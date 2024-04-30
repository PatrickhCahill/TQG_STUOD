# testing gaussianity of psi increment data

from firedrake import *
from utility import Workspace
from tqg.solver import TQGSolver
from firedrake_utility import TorusMeshHierarchy
from firedrake import pi, ensemble,COMM_WORLD, PETSc

from tqg.example2 import TQGExampleTwo as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":
    nx =512
    alpha = 400
    procno = ensemble_comm.rank 
    fname = "/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha if alpha != None else 'none')
    workspace = Workspace(fname)
    fileset = workspace.list_all_h5_files("data", prefix="_pde")
    # gridfname = workspace.output_name("streamfunction_increment_error_grid_value_time_series", "visuals")
    gridfname_reg = workspace.output_name("streamfunction_increment_regularised_error_grid_value_time_series", "visuals")

    if procno == 0:
        grid_pt_id = 0
        grid_pos = {0:"left",1:"middle",2:"right"}
        time_series = np.load(gridfname_reg + ".npy") 
        PETSc.Sys.Print(gridfname_reg, flush=True)
        plotfname = workspace.output_name("psi_inc_res_{}_at_{}.png".format(nx, grid_pos[grid_pt_id]))
        
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
        pltdata = time_series[:,grid_pt_id]
        # pltdata = pltdata[1:] - pltdata[:-1]
        actual_series = np.cumsum(pltdata)
        if False:
            plot_acf(pltdata, lags=100)
            strmfname = workspace.output_name("autocorrelation_streamfunction_res_{}_{}.png".format(nx,grid_pos[grid_pt_id]))
            plt.savefig(strmfname)
        else:
            from hurst import compute_Hc
            series = pltdata
            H, c, data = compute_Hc(series, kind='change', simplified=True)
            # Plot
            f, [ax, ax1, ax2] = plt.subplots(3)
            f.suptitle("H={:.4f}, c={:.4f}".format(H,c))

            ax.plot(data[0], c*data[0]**H, color="deepskyblue")
            ax.scatter(data[0], data[1], color="purple")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Time interval')
            ax.set_ylabel('R/S ratio')
            ax.grid(True)

            ax1.plot(series)
            ax1.set_xlabel('Time interval')
            ax1.set_ylabel('psi increment')
            
            ax2.plot(actual_series)
            
            PETSc.Sys.Print(plotfname)
            plt.savefig(plotfname)
        plt.close()
        

