import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
import numpy as np
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake import * #pi, ensemble,COMM_WORLD, PETSc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqg.example_tqg_theory_paper import TQGTheoryExample as Example1params
#from tqg.example2_tqg_theory_paper import TQGTheoryExampleTwo as Example2params
from test_alpha_convergence_spectrum_control_variables import *

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = 2.5 #opts['time']
    nx = 256#opts['resolution']
    res = nx
    dt = 0.0005 #opts['time_step']
    dump_freq = 40 #opts['dump_freq']
    # alphas = [8.0, 32.0, 180.0, 256.0]
    # alphas = [4.0, 16.0, 64.0, 128.0, 220.0]
    # colours =['b', 'g',  'r',  'm',   'brown']
    # alpha = opts['alpha']
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']

    workspace_none = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, 'none'))

    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="both", quad=True,comm=comm).get_fine_mesh()
    params_none =  Example1params(T,dt,mesh, bc='n', alpha=None)
    solver_none  = TQGSolver(Example1params(T,dt,mesh, bc='n', alpha=None))
    all_h5_files_none = workspace_none.list_all_h5_files(sub_dir='data')

    # flag = 'stream'
    # field_name = {'stream': 'stream', 'buoyancy' : 'buoyancy'}
    # field_name = field_name[flag]
    # field_name = 'buoyancy'
    # field_name = 'stream'
    print(field_name)

    ps_series = {} 
    x_series  = {}  
    legend_series = {} 

    print(len(all_h5_files_none))

    for alpha in alphas:
        # errors = np.zeros(len(all_h5_files))
        legend_series[alpha] = "alpha = {:0.1e}".format(1./alpha**2)
        x_series[alpha] = []
        ps_series[alpha] = []

        for fid in range(76): # range(len(all_h5_files_none)):
            workspace = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, alpha))
            solver_alpha = RTQGSolver(Example1params(T,dt,mesh, bc='n', alpha=alpha))
            all_h5_files = workspace.list_all_h5_files(sub_dir='data')

            assert (len(all_h5_files) == len(all_h5_files_none))

            print("alpha {}, fid {}".format(alpha, fid))

            fname_none = workspace_none.output_name("pde_data_{}.h5".format(fid), "data")
            fname_alpha= workspace.output_name("pde_data_{}.h5".format(fid), "data")
            assert fname_none in all_h5_files_none
            assert fname_alpha in all_h5_files

            field_alpha = None
            field_none = None
            if field_name == 'stream':
                solver_alpha.solve_for_streamfunction_data_from_file(fname_alpha[:-3])
                solver_none.solve_for_streamfunction_data_from_file(fname_alpha[:-3])
                field_alpha = solver_alpha.psi0 if flag_use_regularised_psi else solver_alpha.rpsi0 
                field_none = solver_none.psi0
            elif field_name =='buoyancy' or field_name=='pv':
                solver_alpha.load_initial_conditions_from_file(fname_alpha[:-3])
                solver_none.load_initial_conditions_from_file(fname_none[:-3])
                
                if field_name =='buoyancy':
                    field_alpha = solver_alpha.initial_b
                    field_none = solver_none.initial_b
                if field_name =='pv':
                    field_alpha = solver_alpha.initial_cond
                    field_none = solver_none.initial_cond
            else:
                raise

            error_field = assemble(field_alpha - field_none)

            # setup fourier transform
            xx = np.linspace(0, 1, res+1)
            yy = np.linspace(0, 1, res+1)

            mygrid = []
            for i in range(len(xx)):
                for j in range(len(yy)):
                    mygrid.append([xx[i], yy[j]])
            mygrid = np.asarray(mygrid) # now i have a list of grid values

            grid_values = np.asarray(error_field.at(mygrid, tolerance=1e-10)) 
            grid_values = grid_values.reshape(res+1, res+1).T

            # Fourier analysis working with grid_values
            sampled_values = grid_values[:-1,:-1] # need to get rid off the last point for periodicity
            fourier_transform = 2*np.fft.fft2(sampled_values)
            ps = np.abs(fourier_transform[:int(res/2),:int(res/2)])**2

            k_val_dic = {}
            for i in range(ps.shape[0]):
                for j in range(ps.shape[1]):
                    # ksquared = i**2 + j**2
                    k = np.floor(np.sqrt(i**2 + j**2))
                    # if (ksquared in k_val_dic):
                    if (k in k_val_dic):
                        # k_val_dic[ksquared].append(ps[i,j])
                        k_val_dic[k].append(ps[i,j])
                    else:
                        # k_val_dic[ksquared] = [ps[i,j]]
                        k_val_dic[k] = [ps[i,j]]
            k_val_dic_sorted = sorted(k_val_dic.items(), key = lambda item: item[0])
            x, y = [], []
            for key, val in k_val_dic_sorted:
                #x.append(np.sqrt(key))
                x.append(key)
                y.append(np.mean(val))

            ps_series[alpha].append(y)
            x_series[alpha].append(x)


    for alpha, val in ps_series.items():
        # each val is spectrum data
        dat_ = np.asarray(val)
        print("ps_series: ", alpha, dat_.shape)
        #with open(workspace_none.output_name('powerseries_alpha_{}_field_{}_norotation.npy'.format(alpha,field_name),'data'), "wb") as f:
        #with open(workspace_none.output_name('powerseries_alpha_{}_field_{}.npy'.format(alpha,field_name),'data'), "wb") as f:
        with open(workspace_none.output_name(powerseries_outputname(alpha),'data'), "wb") as f:
            np.save(f, dat_)
    for alpha, val in x_series.items():
        dat_ = np.asarray(val)
        print("x_series: ", alpha, dat_.shape)
        #with open(workspace_none.output_name('wavenumbers_for_powerseries_alpha_{}_field_{}_norotation.npy'.format(alpha,field_name),'data'), 'wb') as f:
        #with open(workspace_none.output_name('wavenumbers_for_powerseries_alpha_{}_field_{}.npy'.format(alpha,field_name),'data'), 'wb') as f:
        with open(workspace_none.output_name(wavenumbers_outputname(alpha),'data'), 'wb') as f:
            np.save(f, dat_)

        

        
        # plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # for i in range(len(legend_series)):
    #     ax.loglog(x_series[i], ps_series[i], label=legend_series[i])
    # ax.grid(True)
    # # ax.legend(loc='upper right')
    # ax.legend(loc='lower left')
    # 
    # plt.title("error {} power spectrums, fig{}".format(field_name,fid))
    # plt.savefig(workspace_none.output_name("{}_error_spectrum_radial_fid_{}.png".format(field_name,fid), "visuals"))
    # plt.close()

    # radial spectrum animation ############################
    #plt.style.use('seaborn-pastel')
    #fig = plt.figure(1)
    #ax = fig.add_subplot(111)
    #ims = [None for alpha in alphas]
    #i = 0
    #ims = []
    #k = 0
    #for alpha in alphas:
    #    im, = ax.loglog([],[], label=legend_series[alpha], alpha=1, c=colours[k])
    #    ims.append(im)
    #    k += 1
    #    # i += 1
    ## im = ax.loglog([], [], [], [])
    #ax.set_title("power spectrum")
    #ax.set(xlabel='log(k)', ylabel='log(E(k))')
    #ax.set_xlim(0., 2e2)
    #ax.set_ylim(1e-23, 1e8)
    #ax.grid(True)
    #ax.legend(loc='lower left')

    #def init():
    #    # im.set_data(np.zeros( (int(res/2), int(res/2)) ) )
    #    # im.set_data([1], [1])
    #    for im in ims: 
    #        im.set_data([1],[1])

    #def animate(i):
    #    print("animate", i)
    #    j = 0
    #    for alpha in alphas:
    #        x = x_series[alpha][i]
    #        y = ps_series[alpha][i]
    #        ims[j].set_data(x, y) # np.log(ps))
    #        j += 1
    #    return ims

    ## plt.savefig("./test_energy_spectrum.jpg")

    #nFrames = len(x_series[alphas[0]])
    #anim = FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=1000)
    #anim.save(workspace_none.output_name('test_spectrum_animation_loglog.gif', "visuals"), writer='imagemagick')
    #plt.close()

    ## radial spectrum animation ############################
    #plt.style.use('seaborn-pastel')
    #fig = plt.figure(1)
    #ax = fig.add_subplot(111)
    #ims = [None for alpha in alphas]
    #i = 0
    #ims = []
    #k = 0
    #for alpha in alphas:
    #    #im, = ax.loglog([],[], label=legend_series[alpha], alpha=1, c=colours[k])
    #    im, = ax.plot([],[], label=legend_series[alpha], alpha=1, c=colours[k])
    #    ims.append(im)
    #    k += 1
    #    # i += 1
    ## im = ax.loglog([], [], [], [])
    #ax.set_title("power spectrum")
    #ax.set(xlabel='k', ylabel='log(E(k))')
    ## ax.set_xlim(0., 2e2)
    #ax.set_ylim(1e-23, 1e8)
    #ax.set_yscale('log')
    #ax.grid(True)
    #ax.legend(loc='lower left')

    #def init():
    #    # im.set_data(np.zeros( (int(res/2), int(res/2)) ) )
    #    # im.set_data([1], [1])
    #    for im in ims: 
    #        im.set_data([1],[1])

    #def animate(i):
    #    print("animate", i)
    #    j = 0
    #    for alpha in alphas:
    #        x = x_series[alpha][i]
    #        y = ps_series[alpha][i]
    #        ims[j].set_data(x, y) # np.log(ps))
    #        j += 1
    #    return ims

    #nFrames = len(x_series[alphas[0]])
    #anim = FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=1000)
    #anim.save(workspace_none.output_name('test_spectrum_animation.gif', "visuals"), writer='imagemagick')
    #plt.close()


    ## plot each wave number as a series #############
    #for wavenumber in alphas:
    #    fig = plt.figure(1)
    #    ax = fig.add_subplot(111)
    #    ax.set_yscale('log')
    #    ax.grid(True)
    #    for alpha in alphas:
    #        x_data = []
    #        y_data = []
    #        for i in range(len(x_series[alpha])):   # number of files 
    #            x_data.append( x_series[alpha][i][wavenumber] )
    #            y_data.append( y_series[alpha][i][wavenumber] )
    #        ax.plot(x_data, y_data, label=legend_series[alpha])

    #    ax.legend(loc='upper left')
    #    plt.title('wavenumber {} power spectrum time series'.format(wavenumber))
    #    plt.savefig(workspace_none.output_name('test_spectrum_time_series_wavenumber_{}.png'.format(wavenumber), "visuals"))
    #    plt.close()

