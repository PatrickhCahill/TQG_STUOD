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
    # alphas = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 180.0, 220.0, 256.0]
    # alphas = [4.0, 16.0, 64.0, 128.0]
    colours =['b', 'g',  'r',  'm',   'brown']
    # alpha = opts['alpha']
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']

    workspace_none = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, 'none'))
    print(field_name)

    # plot each wave number as a series #############
    legend_series = {}
    plt.rcParams.update({"text.usetex": True,"font.family": "serif"})#,"font.sans-serif": ["Helvetica"]})
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.grid(True)

    power_spectrum_alpha_convergence_data = []
    x_alpha_ = []
    _k = 10
    _times = [0.5]
    _t_index = int(0.5 / dt / dump_freq)

    for wavenumber in alphas:
        x_alpha_.append(wavenumber)
        with open(workspace_none.output_name(powerseries_outputname(wavenumber),'data'), "rb") as f:
            y_data = np.load(f) 
        
        with open(workspace_none.output_name(wavenumbers_outputname(wavenumber),'data'), 'rb') as f:
            x_data = np.load(f)  # wavenumber |k|
        #legend_series[wavenumber] = "alpha = {:0.1e}".format(1./wavenumber**2)
        legend_series[wavenumber] = r'$\alpha=1/$' + '{}'.format(wavenumber) + r'$^2$'
        ax.plot(y_data[:, _k], label=legend_series[wavenumber])
        #ax.loglog(x_data[:,10], y_data[:, 10], label=legend_series[wavenumber])
        power_spectrum_alpha_convergence_data.append(y_data[_t_index, _k])

    ax.legend(loc='lower right')
    # plt.title('wavenumber {} power spectrum time series'.format(wavenumber))
    #plt.savefig(workspace_none.output_name('test_spectrum_time_series.png'.format(wavenumber), "visuals"))
    plt.savefig(workspace_none.output_name(spectrum_time_series_plot_outputname(), "visuals"))
    #plt.show()
    plt.close()


    # TEST PLOT alpha convergence in spectral space for a fixed wavenumber
    plt.rcParams.update({"text.usetex": True, "font.family":"serif"}) #,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=18)
    f, ax1 = plt.subplots(1,1,figsize=(8, 6), dpi=300)
    x_values = 1./ np.asarray(alphas)**2 
    x_ = np.zeros(len(alphas))
    # assert np.linalg.norm((x_ - x_values), ord=1) < 0.000000001

    linestyle_tuple = {
         'loosely dotted':        (0, (1, 10)),
         #'dotted':                (0, (1, 1)),
         'densely dotted':        (0, (1, 1)),
         'loosely dashed':        (0, (5, 10)),
         #'dashed':                (0, (5, 5)),
         'densely dashed':        (0, (5, 1)),
         'loosely dashdotted':    (0, (3, 10, 1, 10)),
         'dashdotted':            (0, (3, 5, 1, 5)),
         'densely dashdotted':    (0, (3, 1, 1, 1)),
         'dashdotdotted'      :   (0, (3, 5, 1, 5, 1, 5)),
         'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
         'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
         'solid': 'solid',
         'dotted': 'dotted',
         'dashed': 'dashed',
         'dashdot': 'dashdot'}
    line_styles = {0.3:'dashed', 0.4:'dashdot', 0.5:'dashdotdotted', 0.6:'dotted'}
    marker_styles = {0.3:'x', 0.4:'+', 0.5:'2', 0.6:'^'}
    _legend = {0.3:r't=0.3', 0.4:r't=0.4', 0.5:r't=0.5', 0.6:r't=0.6'}  # 

    for t in _times:
        y_ = np.zeros(x_.shape)
        i = 0  # index for alpha values
        assert len(y_) == len(power_spectrum_alpha_convergence_data)
        for alpha in alphas:
            x_[i] = 1./alpha/alpha  # alpha
            y_[i] = power_spectrum_alpha_convergence_data[i]
            i += 1
        ax1.loglog(x_, y_, linestyle=linestyle_tuple[line_styles[t]], marker=marker_styles[t], color='black', lw=1.3, label=_legend[t])

    # plot reference order 1 alpha
    i = 0
    x_ = np.zeros(len(alphas))
    #y_ = np.zeros(len(alphas))
    for k in alphas: #data.items():
        x_[i] = 1./k/k  # k = alpha # in the equation we have alpha = alpha^2
        i += 1
    ax1.loglog(x_[:], x_[:]*200000, linestyle=linestyle_tuple['densely dashdotted'], color='black', lw=1, label=r'$O(\alpha)$')

    ax1.set_xlabel(r'$\alpha$', labelpad=5)
    ax1.set_ylabel('power spectrum', labelpad=10)
    ax1.grid(True)
    ax1.legend(loc='lower right')
    ax1.title.set_text('power spectrum convergence, '+r'$|k|=${}'.format(_k) + ' ' + 't=0.5')

    plt.savefig(workspace_none.output_name("spectrum_alpha_convergence_{}.png".format(field_name), "visuals"))
    plt.close()


#    for alpha, val in ps_series.items():
#        # each val is spectrum data
#        dat_ = np.asarray(val)
#        print("ps_series: ", alpha, dat_.shape)
#        with open(workspace_none.output_name('powerseries_alpha_{}','data'), "wb") as f:
#            np.save(f, dat_)
#    for alpha, val in x_series.items():
#        dat_ = np.asarray(val)
#        print("x_series: ", alpha, dat_.shape)
#        with open(workspace_none.output_name('wavenumbers_for_powerseries_alpha_{}','data'), 'wb') as f:
#            np.save(f, dat_)
        
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
    ps_series = {}
    x_series = {}
    for wavenumber in alphas:
        #with open(workspace_none.output_name('powerseries_alpha_{}.npy'.format(wavenumber),'data'), "rb") as f:
        with open(workspace_none.output_name(powerseries_outputname(wavenumber),'data'), "rb") as f:
            y_data = np.load(f) 
            ps_series[wavenumber] = y_data
        
        #with open(workspace_none.output_name('wavenumbers_for_powerseries_alpha_{}.npy'.format(wavenumber),'data'), 'rb') as f:
        with open(workspace_none.output_name(wavenumbers_outputname(wavenumber),'data'), 'rb') as f:
            x_data = np.load(f)
            x_series[wavenumber] = x_data
        # print(x_data)

    if 1:
        plt.rcParams.update({"text.usetex": True,"font.family": "serif"})#,"font.sans-serif": ["Helvetica"]})
        # radial spectrum animation ############################
        plt.style.use('seaborn-pastel')
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ims = [None for alpha in alphas]
        i = 0
        ims = []
        k = 0
        for alpha in alphas:
            im, = ax.loglog([],[], label=legend_series[alpha], alpha=1, c=colours[k])
            #im, = ax.plot([],[], label=legend_series[alpha], alpha=1, c=colours[k])
            ims.append(im)
            k += 1
            # i += 1
        # im = ax.loglog([], [], [], [])
        ax.set_title("power spectrum")
        if field_name == 'buoyancy':
            ax.set(xlabel=r'$\ln(k)$', ylabel=r'$\ln(F(b-b^\alpha)^2)$')
            ax.set_ylim(1e-8, 1e12)
        elif field_name == 'stream':
            ax.set(xlabel='k', ylabel=r'$\ln(F(\psi-\psi^\alpha)^2)$')
            ax.set_ylim(1e-23, 1e8)
        elif field_name == 'pv':
            ax.set(xlabel=r'$\ln(k)$', ylabel=r'$\ln(F(\omega-\omega^\alpha)^2)$')
            ax.set_ylim(1e-8, 1e12)
        ax.set_xlim(1,180) 

        # ax.set_ylim(1e-23, 1e8)
        # ax.set_yscale('log')
        ax.grid(True)
        ax.legend(loc='lower left')

        def init():
            # im.set_data(np.zeros( (int(res/2), int(res/2)) ) )
            # im.set_data([1], [1])
            for im in ims: 
                im.set_data([1],[1])

        def animate(i):
            print("animate", i)

            ax.set_title("power spectrum, " + r't={}'.format(i*0.0005*40))
            j = 0
            for alpha in alphas:
                x = x_series[alpha][i]
                y = ps_series[alpha][i]
                ims[j].set_data(x, y) # np.log(ps))
                j += 1
            return ims

        nFrames = 66#len(x_series[alphas[0]])
        anim = FuncAnimation(fig, animate, init_func=init, frames=nFrames, interval=1000)
        #anim.save(workspace_none.output_name('test_spectrum_animation.gif', "visuals"), writer='imagemagick')
        anim.save(workspace_none.output_name(spectrum_animation_outputname(), "visuals"), writer='imagemagick')
        plt.close()



