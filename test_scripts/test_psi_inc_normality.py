from firedrake import *
from utility import Workspace
from scipy import stats
import matplotlib.pyplot as plt

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
    gridfname_reg = workspace.output_name("__streamfunction_increment_regularised_error_grid_value_time_series", "visuals")

    grid_pos = {0:"left",1:"middle",2:"right"}
    if procno == 0:
        time_series = np.load(gridfname_reg + ".npy") 
        PETSc.Sys.Print(gridfname_reg, flush=True)
        plotfname = workspace.output_name("__psi_inc_stats_res_{}.png".format(nx))
        fig, [ax1, ax2, ax3] = plt.subplots(3,3, figsize=(10,7 ))
        
        for ax, grid_pt_id in zip([ax1, ax2, ax3], [0,1,2]):
            pltdata = time_series[:,grid_pt_id]
            actual_series = np.cumsum(pltdata)
            actual_series = actual_series - np.mean(actual_series)
            pltdata = actual_series[1:] - actual_series[:-1]

            ax[0].set_title('Path')
            ax[0].plot(actual_series)

            ax[1].set_ylabel('streamf. inc')
            ax[1].plot(pltdata, linestyle='None', marker='.', markersize=1.)
            shapiro_test = stats.shapiro(pltdata)
            ax[1].set_title('Shap. pv: {:.1e}'.format(shapiro_test.pvalue))

            stats.probplot(pltdata, plot=ax[2])
            ax[2].get_lines()[0].set_markersize(1.)

            plt.subplots_adjust(hspace=.8, wspace=0.4, left=0.08, right=0.95)

        # plt.savefig(plotfname)
        # plt.close()
        plt.show()
