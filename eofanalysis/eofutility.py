#
#   Author: Wei Pan
#   Copyright   2017
#
#   eofanalysis.py
#
#   code for calculating eofs
#

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import time
# import os.path
import warnings
import utility as ut
import pandas as pd
from utilitiesFiredrake import *
from adjustText import adjust_text


# eof_directory = '/home/wpan1/Documents/data/out/eofanalysis/cfl_test/'

# eof_directory = ut.eof_directory
# eof_directory = '/home/wpan1/Documents/data/out/eofanalysis/'


# def gen_delta_x(coarse_filename, fine_filename):
#     warnings.warn("This function is no longer required", DeprecationWarning)
#
#     # fine_traj_filename = "fine_trajectory_" + str(index)
#     # cors_traj_filename = "coarse_trajectory_helmholtz_" + str(index)
#
#     fine = np.genfromtxt(fine_filename + '.csv', delimiter=',')
#     cors = np.genfromtxt(coarse_filename + '.csv', delimiter=',')
#
#     # print "index fine ",  fine
#     # print "index corse ", cors
#     # s = fine.shape[0] * fine.shape[1]
#     # fine = fine.reshape(s)
#     # cors = cors.reshape(s)
#
#     F = fine - cors
#     # F = np.array([np.linalg.norm(F[i:i + 2]) for i in np.arange(0, len(F), 2)])  # try norm of delta_x
#     # F = np.array([complex(F[i], F[i + 1]) for i in np.arange(0, len(F), 2)])   # try complex version
#
#     # return F.reshape(fine.shape[0] * fine.shape[1])  # serialise delta_x_ij to a serial array
#     return F[:, 0], F[:, 1]   # return x and y components


# def correct(a):
#     a[a < 0] = 0
#     # for i in range(len(a)):
#     #     if a[i] < 0:
#     #         a[i] = 0


# def test_eofs():
#     warnings.warn("This function is no longer required", DeprecationWarning)
#
#     eig_vals = np.genfromtxt(eof_directory + "eig_vals.csv", delimiter=",")
#     eofs = np.genfromtxt(eof_directory + "eofs.csv", delimiter=",")
#     print(eig_vals.shape, eofs.shape)


# class EOFUtility:

    # def __init__(self):
        # no index
        # self.coarse_filename = coarse_traj_filename
        # self.fine_filename = fine_traj_filename
        # self.number_of_eofs = []
        # self.vars = [0.7, 0.8, 0.9, 0.95]
        # self.ndays = 365
        # self.dxndays = 1.0

    # def compute_eofs(self):
    #     warnings.warn("deprecated, use new eofsolver", DeprecationWarning)
    #     vars = self.vars
    #     ndays = self.ndays
    #     dx_ndays = self.dxndays
    #
    #     grid_pts = Function(vfs_c).interpolate(coord_c).dat.data[:]
    #
    #     F = np.zeros((ndays, 4225, 2))
    #
    #     for index in range(ndays):
    #         dxname = ut.trajectory_dir + "delta_x_{}_at_t_{}.csv".format(index * 4, dx_ndays)  # todo 1.0 for 1 day
    #         value = pd.read_csv(dxname, names=['x', 'y'])
    #         F[index, :, 0] += value['x']
    #         F[index, :, 1] += value['y']
    #
    #     # reshape F into data matrix
    #     data = np.reshape(F, (ndays, 4225*2))
    #     vals, eofs = self.eof_solver(data)
    #
    #     perm = vals.argsort()  # sort according to abs of eigenvalues
    #     perm = perm[::-1]  # descending
    #     vals = vals[perm]
    #     normalised_vals = vals / np.sum(vals)
    #     # compute the number of eofs such that we obtain 90% of variance
    #     # var = 0.9
    #     num_eofs = self.show_number_of_eofs(normalised_vals, vars)
    #
    #     # print(self.coarse_filename, num_eofs)
    #     # self.number_of_eofs = num_eofs
    #
    #     self.show_spectrum(vals, num_eofs)
    #     self.show_normalised_spectrum(normalised_vals, num_eofs)
    #
    #     # eofs = eofs[perm]
    #     #
    #     # # print "lambda shape ", eig_vals_x.shape  # each one should be x corresponding to x at each (i,j) in the domain
    #     # # print "eofs shape", eofs_x.shape
    #     #
    #     # correct(vals)
    #     #
    #     # index = 0
    #     # for _v in var:
    #     #     self.write_to_file(eig_vals_x, eofs_x, num_eofs[index], lambda_output_name_x + str(_v),
    #     #                        eof_output_name_x + str(_v))
    #     #     self.write_to_file(eig_vals_y, eofs_y, num_eofs[index], lambda_output_name_y + str(_v),
    #     #                        eof_output_name_y + str(_v))
    #     #     index += 1


    # def generate_eofs_decoupled(self, lambdaxname, lambdayname, xixname, xiyname):
    #     warnings.warn("This function is no longer required", DeprecationWarning)
    #     var = self.vars
    #
    #     ndays = self.ndays
    #     dx_ndays = self.dxndays
    #     Fx = np.zeros((ndays, 4225))  # (days x grid points)
    #     Fy = np.zeros((ndays, 4225))
    #     for index in range(ndays):
    #         dxname = ut.trajectory_dir + "delta_x_s_{}_at_t_{}.csv".format(index * 4, dx_ndays)  # todo 1.0 for 1 day
    #         value = pd.read_csv(dxname, names=['x', 'y'])
    #         Fx[index, :] += value['x']
    #         Fy[index, :] += value['y']
    #
    #     eig_vals_x, eofs_x = self.eof_solver(Fx)
    #     eig_vals_y, eofs_y = self.eof_solver(Fy)
    #
    #     correct(eig_vals_x)
    #     correct(eig_vals_y)
    #
    #     # x
    #     permx = eig_vals_x.argsort()  # sort according to abs of eigenvalues
    #     permx = permx[::-1]           # descending
    #     eig_val_x = eig_vals_x[permx]
    #     eofs_x = eofs_x[:, permx]
    #     normalised_eigv_x = eig_val_x / np.sum(eig_val_x)
    #
    #     num_eofs_x = self.show_number_of_eofs(normalised_eigv_x, var)
    #     self.show_normalised_spectrum(normalised_eigv_x, num_eofs_x, 'decoupled/x_')
    #
    #     # y
    #     permy = eig_vals_y.argsort()  # sort according to abs of eigenvalues
    #     permy = permy[::-1]           # descending
    #     eig_val_y = eig_vals_y[permy]
    #     eofs_y = eofs_y[:, permy]
    #     normalised_eigv_y = eig_val_y / np.sum(eig_val_y)
    #
    #     num_eofs_y = self.show_number_of_eofs(normalised_eigv_y, var)
    #     self.show_normalised_spectrum(normalised_eigv_y, num_eofs_y, 'decoupled/y_')
    #
    #     # print "lambda shape ", eig_vals_x.shape  # each one should be x corresponding to x at each (i,j) in the domain
    #     # print "eofs shape", eofs_x.shape
    #
    #     # index = 0
    #     # for _v in var:
    #     #     self.write_to_file(eig_vals_x, eofs_x, num_eofs[index], lambda_output_name_x + str(_v),
    #     #                        eof_output_name_x + str(_v))
    #     #     self.write_to_file(eig_vals_y, eofs_y, num_eofs[index], lambda_output_name_y + str(_v),
    #     #                        eof_output_name_y + str(_v))
    #     #     index += 1


    # def generate_eofs(self, lambda_output_name_x, lambda_output_name_y,
    #                   eof_output_name_x, eof_output_name_y):
    #     warnings.warn("deprecated, use new eofsolver", DeprecationWarning)
    #     var = self.vars
    #
    #     # read in data from csv files
    #     # Fx = np.array([])
    #     # Fy = np.array([])
    #     ndays = self.ndays
    #     dx_ndays = self.dxndays
    #     # F = np.zeros((ndays, 4225, 2))
    #     Fx = np.zeros((ndays, 4225))  # (days x grid points)
    #     Fy = np.zeros((ndays, 4225))
    #     # start_time = time.time()
    #     for index in range(ndays):
    #         # i = 2 * index
    #         # if os.path.isfile(eof_directory + "fine_trajectory_" + str(i) + ".csv"):
    #         # not sure if this works
    #         dxname = ut.trajectory_dir + "delta_x_s_{}_at_t_{}.csv".format(index * 4, dx_ndays)  # todo 1.0 for 1 day
    #         value = pd.read_csv(dxname, names=['x', 'y'])
    #         # F[index, :, 0] += value['x']
    #         # F[index, :, 1] += value['y']
    #         Fx[index, :] += value['x']
    #         Fy[index, :] += value['y']
    #         # cors_filename = self.coarse_filename + str(i)
    #         # fine_filename = self.fine_filename + str(i)
    #         # if i == 0:
    #         #     Fx, Fy = gen_delta_x(cors_filename, fine_filename)
    #         # else:
    #         #     tempx, tempy = gen_delta_x(cors_filename, fine_filename)
    #         #     Fx = np.vstack([Fx, tempx])     # very slow, if using numpy array better to set the whole thing first
    #         #     Fy = np.vstack([Fy, tempy])
    #
    #     # print Fx.shape, Fy.shape
    #
    #     # print("----%s seconds------" % (time.time() - start_time))
    #     eig_vals_x, eofs_x = self.eof_solver(Fx)
    #     eig_vals_y, eofs_y = self.eof_solver(Fy)
    #
    #     # print eig_vals_x
    #
    #     # form abs matrix
    #     eigv_abs = np.array([np.linalg.norm([eig_vals_x[i], eig_vals_y[i]]) for i in range(len(eig_vals_x))])
    #     perm = eigv_abs.argsort()  # sort according to abs of eigenvalues
    #     perm = perm[::-1]  # descending
    #     eigv_abs = eigv_abs[perm]
    #     normalised_eigv_abs = eigv_abs / np.sum(eigv_abs)
    #     # compute the number of eofs such that we obtain 90% of variance
    #     # var = 0.9
    #     num_eofs = self.show_number_of_eofs(normalised_eigv_abs, var)
    #
    #     # print(self.coarse_filename, num_eofs)
    #     # self.number_of_eofs = num_eofs
    #
    #     self.show_spectrum(eigv_abs, num_eofs)
    #     self.show_normalised_spectrum(normalised_eigv_abs, num_eofs)
    #
    #     eig_vals_x = eig_vals_x[perm]
    #     eig_vals_y = eig_vals_y[perm]
    #     eofs_x = eofs_x[:, perm]
    #     eofs_y = eofs_y[:, perm]
    #
    #     # print "lambda shape ", eig_vals_x.shape  # each one should be x corresponding to x at each (i,j) in the domain
    #     # print "eofs shape", eofs_x.shape
    #
    #     correct(eig_vals_x)
    #     correct(eig_vals_y)
    #
    #     index = 0
    #     for _v in var:
    #         self.write_to_file(eig_vals_x, eofs_x, num_eofs[index], lambda_output_name_x + str(_v), eof_output_name_x + str(_v))
    #         self.write_to_file(eig_vals_y, eofs_y, num_eofs[index], lambda_output_name_y + str(_v), eof_output_name_y + str(_v))
    #         index += 1
    #     # self.write_to_file(eig_vals_x, eofs_x, num_eofs, "eig_vals_x" + str(var), "eofs_x" + str(var))
    #     # self.write_to_file(eig_vals_y, eofs_y, num_eofs, "eig_vals_y" + str(var), "eofs_y" + str(var))

    # # F is the matrix with columns being data time series
    # def eof_solver(self, data):
    #     warnings.warn("deprecated, use new eofsolver", DeprecationWarning)
    #     n = data.shape[0]  # should be number of days
    #
    #     assert n - self.ndays < 0.000001
    #
    #     mean = data.mean(axis=0)
    #     F = data - mean[None, :]
    #     R = F.T.dot(F)
    #     # print(R.shape[0])
    #     # R /= -1. + R.shape[0]  # unbiased estimator this is incorrect!
    #     R /= -1. + n
    #
    #     # print R.shape[0]
    #
    #     # covariance matrix is symmetric R^T = R
    #     # so by Spectral Theorem the eigenvalues are real and the eigenvectors are orthonormal
    #
    #     # eigenvalues are in ascending order
    #     # corresponding eigen vectors are in the same order
    #     # eig_vals[i] corresponds to eofs[:,i]
    #     eig_vals, eofs = np.linalg.eig(R)
    #
    #     # print eig_vals
    #     eig_vals = eig_vals.real
    #     eofs = eofs.real
    #
    #     # # sanity check ###
    #     # eig_vecs should be unity
    #     # eig_vals should be positive
    #     for val in eig_vals:
    #         assert val >= -1.e-15
    #     v = np.sum(abs(eofs**2), axis=0)
    #     assert np.sum(v) - v.size < .0000001
    #
    #     return eig_vals, eofs


# def show_data(data, nums, label, added_dir='', outputname='normalised_spec_plot'):
#     out_dir = ut.plot_dir + added_dir
#     # test_data = np.exp(nums)
#     # data = test_data
#
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     colors = ['c', 'm', 'r']
#     labels = [r'$n_{\bf{\xi}} \equiv 50\%$',
#               r'$n_{\bf{\xi}} \equiv 70\%$',
#               r'$n_{\bf{\xi}} \equiv 90\%$']
#
#     ax.plot(data)
#     for i in range(len(nums)):
#         ax.scatter(nums[i], data[nums[i]], marker='o', color=colors[i], label=labels[i])
#
#     # annotation
#     annotations = []
#     # for xy in zip(nums[2:], data[nums[2:]]):
#     #     ax.annotate('%s' % xy[0], xy=xy, textcoords='data')
#
#     for i, txt in enumerate(nums):
#         annotations.append(ax.text(nums[i], data[nums[i]], txt))
#     adjust_text(annotations)
#
#     plt.ylabel(label, fontsize=12)
#     plt.xlabel('number of EOFs', fontsize=12)
#     plt.xlim([-10, 250])
#     plt.legend()
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig("{}{}_{}.svg".format(out_dir, ut.scheme, outputname), format='svg', dpi=1200)
#
#     # def show_spectrum(self, eig_vals, nums, added_dir=''):
#     #     self.show_data(eig_vals, nums, "spectrum", added_dir)
#     #
#     # def show_normalised_spectrum(self, normalised_eig_vals, nums, added_dir=''):
#     #     self.show_data(normalised_eig_vals, nums, "normalised spectrum", added_dir)
#
#     # number of eofs that capture the target amount of variation
#     # target_percentage must be < 1
#
#
# def show_number_of_eofs(normalised_eig_vals, target_percentages):
#     result = []
#     for target_percentage in target_percentages:
#         assert target_percentage < 1.000000001
#         i = 0
#         percent = 0
#         while percent < target_percentage:
#             percent += normalised_eig_vals[i]
#             i += 1
#         result.append(i)
#     return result

    # ## truncate to N eofs
    # def eof_truncation(self, N):
    #     self.pc_trunc = self.F.dot(self.eofs[:,0:N])

    # def write_to_file(self, eig_vals, eofs, num_eofs, eigv_filename, eof_filename):
    #     warnings.warn("deprecated. no longer separate x and y axis", DeprecationWarning)
    #     print(num_eofs)
    #     np.savetxt(eof_directory + eigv_filename + ".csv", eig_vals[:num_eofs], delimiter=",")
    #     # np.savetxt(eof_directory + "normalised_eig_vals.csv", normalised_eig_vals[:num_eofs], delimiter=",")
    #     np.savetxt(eof_directory + eof_filename + ".csv", eofs[:, :num_eofs], delimiter=",")
