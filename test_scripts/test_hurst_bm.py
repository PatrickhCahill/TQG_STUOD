import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc, random_walk
from scipy.ndimage import gaussian_filter1d

# Use random_walk() function or generate a random walk series manually:
# series = random_walk(99999, cumprod=True)
np.random.seed(42)
# random_changes = 1. + np.random.randn(99999) / 1000.
# series = np.cumprod(random_changes)  # create a random walk from random changes
dt = 0.001
T = 1.
N = int(T/dt)

bm_incs = np.random.normal(0,np.sqrt(dt), N)
bm = np.cumsum(bm_incs)
sigma = 0.20
r = 1

gbm_incs = np.exp((r - 0.5*sigma**2)*dt + sigma * bm_incs) 
gbm = np.cumprod(gbm_incs)

series = np.log(gbm_incs) - (r - 0.5*sigma**2)*dt# - (r- 0.5*sigma**2)*np.cumsum( dt + np.zeros(gbm.shape) )
print(series.shape)

series = gaussian_filter1d(bm_incs,4)
gf_bm = np.cumsum(series)

# Evaluate Hurst equation
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
ax2.plot(gf_bm)
plt.show()
