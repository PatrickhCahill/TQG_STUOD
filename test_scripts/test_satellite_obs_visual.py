from relative_import_header import *
from mpl_toolkits import mplot3d
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from celluloid import Camera
from utility import Workspace

wspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

cdata = np.load(wspace.output_name("obs_data.npz", "cPDESolution"))
fdata = np.load(wspace.output_name("obs_data.npz", "PDESolution" ))
sub_obs_site_indices = np.load(wspace.output_name("sub_obs_site_indices.npy", "ParamFiles"))

print(len(cdata))
print(len(fdata))

print(type(cdata['obs_data_0']))
print(cdata['obs_data_0'].shape)

if 1:
    #plt.rc('xtick', labelsize=4)
    #plt.rc('ytick', labelsize=4)
    #plt.rc('axes',  labelsize=3)
    #plt.rc('font', size=3)
    fig = plt.figure(figsize=(8,4)) #subplots((2,1))

    fig.tight_layout()
    #fig.patch.set_facecolor('orange')
    #fig.patch.set_alpha(0.0)
    #fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # subplot 1 ###########
    #ax = plt.axes()
    #ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,1)
    ax2.set_aspect(aspect='equal')
    #ax2.set_ylabel('y-axis')
    #ax2.set_xlabel('x-axis')
    #ax2.grid(True)
    #ax2.patch.set_facecolor('orange')
    #ax2.patch.set_alpha(0.5)
    ax2.axis('on')
    ax2.grid(False)
    ax2.set_xlabel("x-axis")
    ax2.set_ylabel("y-axis")
    ax2.set_title('coarse res obs')


    ax1 = fig.add_subplot(1,2,2)
    ax1.set_aspect(aspect='equal')
    ax1.axis('on')
    ax1.grid(False)
    ax1.set_xlabel("x-axis")
    ax1.set_ylabel("y-axis")
    ax1.set_title('fine res obs')

    camera = Camera(fig)

    r = 1./2./np.pi
    number_of_tracks = 4
    number_of_tracks2 = 5
    number_of_track_samples = 900 
    total_theta  = 2*np.pi* number_of_tracks
    total_theta2 = 2*np.pi* number_of_tracks2

    k_tilde = number_of_tracks/number_of_tracks2
    k_sum = number_of_tracks2 + number_of_tracks

    ms = np.linspace(0, number_of_track_samples, number_of_track_samples, endpoint=False)
    ns = np.linspace(0, k_sum, k_sum, endpoint=False) 

    theta2 = total_theta2 / number_of_track_samples * ms
    theta = k_tilde*theta2

    x2 = theta2 / total_theta2
    x = theta/total_theta

    dump_freq = 10
    delta_t = 0.0001
    delta_t_per_save = dump_freq * delta_t  
    delta_t_per_day = 411 * delta_t
    delta_t_per_10_days = 4110 * delta_t

    number_of_saves = 1501
    total_time = number_of_saves * delta_t_per_save

    time = np.linspace(0, number_of_saves, number_of_saves, endpoint=False)

    intersections = 2*np.pi/(k_tilde+1)*ns

    scaling = number_of_track_samples / k_sum
    ns *= scaling
    ns = np.setdiff1d(ns, np.delete(ns, np.where(np.mod(ns,1)==0))).astype(int)
    ms_2 = np.setdiff1d(ms, ns)

    theta2_removed_duplicates = total_theta2/number_of_track_samples * ms_2
    x2_removed_duplicates = theta2_removed_duplicates / total_theta2

    for t in time:
        #print("time ", t)
        #c = t * 2 * np.pi 
        c = t * 2*np.pi * delta_t_per_save / delta_t_per_10_days
        z = r * np.sin(theta + c)
        y = r * np.cos(theta + c)
        
        z2 = r * np.sin(-theta2_removed_duplicates + c)
        y2 = r * np.cos(-theta2_removed_duplicates + c)


        z3 = r * np.sin(intersections * k_tilde + c)
        y3 = r * np.cos(intersections * k_tilde + c)


        # subplot 2 ###########

        _x = x * total_theta 
        y  = np.arccos(z/r) *  np.sign(y) * r + 0.5
        y_ = np.arccos(z2/r) * np.sign(y2) * r + 0.5
        y3 = np.arccos(z3/r) * np.sign(y3) * r + 0.5
        #ax2.scatter( y, _x / total_theta,  s=0.5, color='lime')
        #ax2.scatter( y_, _x /total_theta, s=0.5, color='deeppink')

        #ax2.scatter( x, y,  s=15.2, marker='+', color='lime')
        #ax2.scatter( x2_removed_duplicates, y_,  s=15.2, marker='x', color='deeppink')
        #ax2.scatter(intersections / total_theta2, y3, s=1.5, color='blue')
        
        #print(x, type(x))
        #print(y, type(y))

        combine_xs = np.concatenate((x, x2_removed_duplicates))
        combine_ys = np.concatenate((y, y_))

        combined = np.vstack((combine_xs, combine_ys)).T

        #ax2.scatter(combined[:,0], combined[:,1], color='blue', s=0.1)
        #print(combined, combined.shape, )
        #ax1.scatter(combined[:,1], 

        def _plot(ax, dat, xs):
            #xx, yy = np.meshgrid(xs[:,0], xs[:,1])
            #zz = griddata(combined, dat, (xx, yy)) #, method='nearest')
            #ax.pcolor(zz, cmap='turbo', vmin=-1, vmax=1)
            ax.scatter(xs[:, 0], xs[:,1], c=dat, cmap='turbo', vmin=-.5, vmax=0.5, s=0.2)

        _plot(ax2, cdata['obs_data_{}'.format(int(t))][sub_obs_site_indices], combined[sub_obs_site_indices,:])
        _plot(ax1, fdata['obs_data_{}'.format(int(t))][sub_obs_site_indices], combined[sub_obs_site_indices,:])

        #ax1.pcolormesh(combined[:, 0], combined[:,1], fdata['obs_data_{}'.format(int(t))], cmap='turbo', vmin=-1, vmax=1)
        #ax2.pcolormesh(combined[:, 0], combined[:,1], cdata['obs_data_{}'.format(int(t))], cmap='turbo', vmin=-1, vmax=1)
        
        camera.snap()



    animation = camera.animate(blit=False, interval=200)  # interval is in units of milliseconds or 1e-3 seconds
    animation.save('./test_view_satellite_obs_reduced.mov', codec='png', savefig_kwargs={'transparent':False, 'facecolor':'white'}, dpi=100)

