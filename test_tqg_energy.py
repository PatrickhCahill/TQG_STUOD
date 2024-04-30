import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake import pi, ensemble,COMM_WORLD, PETSc

from tqg.example2 import TQGExampleTwo as Example2params
# from tqg.example2dp import TQGExampleTwoDoublyPeriodic as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = opts['time']
    nx = opts['resolution']
    dt = opts['time_step']
    dump_freq = opts['dump_freq']
    alpha = opts['alpha']
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']

    PETSc.Sys.Print("time horizon: {}, res: {}, dt: {}, dump_req: {}, alpha: {}".format(T, nx, dt, dump_freq, alpha), flush=True)

    workspace = Workspace("/home/wpan1/Data2/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha if alpha != None else 'none'))
    
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=comm).get_fine_mesh()
    
    params = Example2params(T,dt,mesh, bc='x', alpha=alpha)

    solver = RTQGSolver(params) if alpha!=None else TQGSolver(params)

    visual_output_name = workspace.output_name("_pde_visuals", "visuals")
    data_output_name   = workspace.output_name("_pde_data", "data")
    # h5_data_name = workspace.output_name("_pde_data_3275", "data")
    # solver.load_initial_conditions_from_file(h5_data_name)
    kinetic_energy_series, potential_energy_series, total_energy_series, casimir_series, non_casimir_series = solver.solve(dump_freq, visual_output_name, data_output_name, ensemble_comm, do_save_data=do_save_data, do_save_visual=do_save_visual, do_save_spectrum=do_save_spectrum, res=nx)
    
    if ensemble_comm.rank == 0:
        import matplotlib.pyplot as plt
        f, [ax1, ax2, ax3] = plt.subplots(3)

        ax1.plot(potential_energy_series, label='potential energy', c='r')
        ax1.plot(kinetic_energy_series, label='kinetic energy', c='b')
        ax1.plot(total_energy_series, label='total energy', c='g')

        ax1.set_xlabel('time steps')
        ax1.set_ylabel('energy')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2.plot(casimir_series, label='casimir', c='m')
        ax2.set_xlabel('time steps')
        ax2.set_ylabel('casimir')
        ax2.grid(True)
        ax2.legend(loc='upper left')

        ax3.plot(non_casimir_series, label='non casimir', c='m')
        ax3.set_xlabel('time steps')
        ax3.set_ylabel('non casimir')
        ax3.grid(True)
        ax3.legend(loc='upper left')

        # plt.savefig("/home/wpan1/Documents/Data/tqg_example3/tqg_" + "all_energies_0_{}.png".format(step))
        plt.savefig(workspace.output_name("energy_plot.png", "visuals"))
        plt.close()

