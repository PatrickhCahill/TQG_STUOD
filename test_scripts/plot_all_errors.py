import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import warnings

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))

# load data ####
#names = {'64':"_64", '64sdt': "_64_smaller_dt", '128lim': "_128_limiter", '128':"_128", '128sdt':"_128_smaller_dt"}
#names = ['64', 'l_64_smaller_dt','64_smaller_dt', '128_limiter', '128', '128_smaller_dt']
names = ['64', '64_smaller_dt']
legend_names = ['64, 1e-2', '64, 1e-4']
# names =['l_64_smaller_dt']
colors = ['b', 'r', 'g', 'y', 'm', 'aqua']
datadir = '/home/wpan1/Data/PythonProjects/tqg_example3_'

legendhandls =[]
for n,_n, co in zip(names,legend_names, colors):
    fname = lambda v : datadir + n + '/visuals/tqg_test{}_errors.npy'.format(v)
    try:
        l1 = ax1.semilogy(np.load(fname('q'))[:],c=co, label=_n)
        ax2.semilogy(np.load(fname('b'))[:],c=co, label=_n)
        print("loaded ", fname('q'))
        legendhandls.append( mlines.Line2D([],[], color=co, marker='*',label=_n))
    except Exception as e:
        print("failed to load data: '%s'" %e)

ax1.set_title('PV')
ax2.set_title('Buoyancy')
ax1.set_xlabel('time (in 0.1 time units)')
ax2.set_xlabel('time (in 0.1 time units)')
ax1.set_ylabel('Num Solution vs Initial Cond, L1 norm')
fig.legend(handles=legendhandls,loc='center right')
plt.subplots_adjust(left=0.09,right=0.85)
plt.savefig('./all_errors.png', dpi=100)
plt.close()

