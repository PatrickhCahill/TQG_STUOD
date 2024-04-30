# This script tries to plot the growth rate (imaginary component) of the dispersion relation

import numpy as np

def quartic(alpha, k):
    assert(not (alpha < 0))
    assert(k >=0)

    return alpha * k**4 + (1 + alpha)*k**2 + 1

def growth_rate(k, alpha=2.):
    return 1./quartic(alpha, k) * np.sqrt(quartic(alpha, k) - 1)

def decay_rate(k, alpha):
    return np.sqrt(1./quartic(alpha, k))

alphas = [0., 0.1, 1., 4.]
line_style = {0.:'solid', 0.1:'dashed', 1.:'dotted', 4.:'dashdot'}

import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots(1)

x = np.arange(0.000, 9.000, 0.01)

for a in alphas:
    y = [growth_rate(k, a) for k in x]
    ax.plot(x, y, label=r'$\alpha = {}$'.format(a), linestyle=line_style[a], color='k')
    # y = [decay_rate(k, a) for k in x]
    # ax[1].plot(x, y, label=r'$\alpha = {}$'.format(a))

ax.legend(loc='upper right', fontsize=17)
ax.set_xlabel(r'wavenumber $|{\bf k}|$',fontsize=18, labelpad=5)
ax.set_ylabel('growth rate', fontsize=18, labelpad=10)
ax.set_xlim([0,5])
fig.tight_layout()


# ax[1].legend(loc='upper right', fontsize=12)

plt.savefig("./growth_rate.png")



