import numpy as np
from scipy.ndimage import gaussian_filter1d as gfilter
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

dt = 0.001
T = 1.
N = int(T/dt)

bm_incs = np.random.normal(0, np.sqrt(dt), N)
lognorms = stats.lognorm.rvs(0.954, size=N)
bm = np.cumsum(bm_incs)
gf_bm_incs = gfilter(bm_incs, 3)

fig, [[ax1, ax2],[ax3,ax4]] = plt.subplots(2,2)

# ax1.plot(bm)
ax1.set_ylabel('BM incs')
# ax1.plot(np.cumsum(gf_bm_incs))
ax1.plot(bm_incs, linestyle='None', marker='.')
test_bm = stats.shapiro(bm_incs)
ax1.set_title('shapiro pval: {:.1e}'.format(test_bm.pvalue))
# ax2.plot(bm_incs)
# ax2.set_ylabel('BM inc')

# ax2.hist(bm_incs, density=True, bins=50)
stats.probplot(bm_incs, plot=ax3)

ax2.plot(gf_bm_incs, linestyle='None', marker='.')
ax2.set_ylabel('gf BM incs')
test_fbm = stats.shapiro(gf_bm_incs)
ax2.set_title('shapiro pval: {:.1e}'.format(test_bm.pvalue))
# ax3.set_ylabel('reg. BM inc')
# ax2.hist(gf_bm_incs, density=True, bins=20)
stats.probplot(gf_bm_incs, plot=ax4)

plt.subplots_adjust(hspace=.5, wspace=0.5)
plt.show()
