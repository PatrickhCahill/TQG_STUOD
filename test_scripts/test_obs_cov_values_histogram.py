# 
# load satellite track observation data (of fine and coarse resolutions)
# compute covariance matrix
# the method here is like the eof computation
#

from relative_import_header import *
from data_assimilation.data_assimilation_utilities import data_assimilation_workspace as da_workspace
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import ldl


def plot(_values, lims_low, lims_high, file_name):
    HIST_BINS = np.linspace(_values.min(), _values.max(), 100)
    print(HIST_BINS)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(_values, HIST_BINS, lw=2)
    #ax.set_xlim(lims_low, lims_high)
    ax.set_xlim(lims_low, 2e-12)
    #ax.set_ylim(top=10)


    plt.savefig(wspace.output_name(file_name, 'ParamFiles'))

wspace = da_workspace('/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin')

cov_matrix = np.load(wspace.output_name('obs_cov.npy', 'ParamFiles'))
#print(np.linalg.matrix_rank(cov_matrix))
rank = np.linalg.matrix_rank(cov_matrix)

lu, d, perm = ldl(cov_matrix, lower=True)

#np.savetxt(wspace.output_name("obs_cov_ldl_d.csv", 'ParamFiles'), d, delimiter=',')

#u, s, vh = np.linalg.svd(cov_matrix, hermitian=True)
#print(perm)

#d[d<1e-12] = 0
d_serial = np.diag(d)
rank = len(d_serial[d_serial > 1e-14])
print("rank:", rank)
#print(d_serial[:rank].max(), d_serial[:rank].min())

#_A = np.matmul(lu, d)
#_A = np.matmul(_A, lu.T)
#print( np.linalg.norm(_A - cov_matrix) )
#print(d)
#print(perm)

perm_inv = np.zeros(perm.shape, dtype=np.int64)
for i in range(len(perm)):
    perm_inv[perm[i]] = i

P = np.eye(len(perm))[perm,:]

cov_sub_matrix = P.dot(cov_matrix).dot(P.T)[:rank, :rank]
print(cov_sub_matrix.shape)
cov_sub_matrix_inv = np.linalg.inv(cov_sub_matrix)

np.save(wspace.output_name('obs_cov_sub_matrix_inv', 'ParamFiles'), cov_sub_matrix_inv)
np.save(wspace.output_name('sub_obs_site_indices', 'ParamFiles'), perm[:rank])
_proper_indices = np.arange(0,rank)
_diff = perm[:rank] - _proper_indices
print(_diff[_diff > 0])
#print(perm_inv[:rank])
np.savetxt(wspace.output_name('sub_obs_site_indices.csv', 'ParamFiles'), perm[:rank], delimiter=',')
#lu_permed = lu[perm,:]
#_A = np.matmul(lu_permed, d)
#_A = _A[perm_inv, :]
#_A = _A.dot(lu.T)
#print(np.linalg.norm(_A - cov_matrix))

if 0:
    d_ = np.copy(d)

    sum_ = 0
    num_off_diag = 0
    _sum_max = 0    # maximum nonzero value in the off diag
    _sum_min = 100  # minimum nonzero value in the off diag
    _sum_max_ij = [0,0]

    for i in range(len(d_)):
        for j in range(len(d_)):
            if not (i == j):
                sum_ += d_[i,j]
                num_off_diag +=1

                if d_[i,j] > _sum_max:
                    _sum_max_ij = [i, j]
                    _sum_max = d_[i,j]

                if d_[i,j] > 0:
                    _sum_min = min(d_[i,j], _sum_min)
            else:
                pass

    print("sum_:", sum_)
    print("_sum_max, _sum_min:",_sum_max, _sum_min)
    print("_sum_max and ij: ", _sum_max, _sum_max_ij)
    print(d_[_sum_max_ij[0], _sum_max_ij[1]], _sum_max_ij[0], _sum_max_ij[1])

if 0:
    print(cov_matrix.dtype)
    print(d_.dtype)
    #thresh = d_.max() * max(d_.shape)*np.finfo(cov_matrix.dtype).eps
    #thresh_svd = s.max() * max(cov_matrix.shape)*np.finfo(cov_matrix.dtype).eps
    print(thresh, thresh_svd)
    num_nonzeros = len(d_[d_ > thresh_svd])
    print(num_nonzeros)
    print(len(s) - len(s[s < thresh_svd]))

#d_diag = np.diag(d)
#
#plot_values = d_diag[d_diag>0]
#plot_values = plot_values[plot_values < 1e-11]
#
#plot(plot_values, plot_values.min(), plot_values.max(), 'd_diag_histogram.png') 


#size = len(cov_matrix)
##size = 5
#
#sub = np.copy(cov_matrix[:size, :size])
#print(np.linalg.matrix_rank(sub, tol=1e-10)) #, hermitian=True))
#
#diag = cov_matrix.diagonal()
#
#for i in range(len(sub)):
#    sub[i,i] = 0
#
#lower_triangle = np.tril(sub)
#
#off_diag = lower_triangle.reshape(size**2)
##print(lower_triangle.reshape(5*5))
#off_diag = off_diag[off_diag!=0.]
#print('off_diag', '[',off_diag.min(),off_diag.max(), ']')
##print(off_diag)
#print(off_diag.shape)
#print('diag    ', '[',diag.min(), diag.max(),']')
#print(diag.shape)
##
##
#
#plot(off_diag, -1, 1, 'obs_cov_histogram.png')
##plot(diag, 0, 5, 'obs_var_histogram.png')
