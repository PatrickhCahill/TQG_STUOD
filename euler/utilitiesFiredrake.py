# import numpy as np
from firedrake import *

import utility
# from pde import euler

dt = 0.0025  # 3.6 minutes
# dt_coarse = 0.02   # 28.8 minutes
# T = 365.
L = 1.
n0 = 8
mesh_level = 7                    # 8[0]   16[1]   32[2]   64[3]   128[4]   256[5]   512[6]
mesh_truth_index = mesh_level - 1  # 512x512
# mesh_signal_index = 3             # 64x64
# mesh_obs_index = 1  # 16x16
mesh_base = SquareMesh(n0, n0, L)
#
# MeshHierarchy creates levels of meshes of different refinements in increasingly fine order
# i.e. mesh_hierarchy[0] is the coarsest, mesh_hierarchy[1] is the second coarsest
mesh_hierarchy = MeshHierarchy(mesh_base, mesh_truth_index)
# dumpfreq = 200  # corresponds to half a day 0.5/dt
# dumpfreq_coarse = 25
#
fine_msh = mesh_hierarchy[mesh_truth_index]

# coarse_msh = mesh_hierarchy[mesh_signal_index]
# obs_msh = mesh_hierarchy[mesh_obs_index]
# fine_dg_space = FunctionSpace(fine_msh, "DG", 1)
# coarse_dg_space = FunctionSpace(coarse_msh, "DG", 1)
# obs_dg_space = FunctionSpace(obs_msh, "DG", 1)
# obs_vfs = VectorFunctionSpace(obs_msh, "CG", 1)
# vfs_c = VectorFunctionSpace(coarse_msh, "CG", 1)
# vfs_f = VectorFunctionSpace(fine_msh, "CG", 1)
# coord_f = SpatialCoordinate(vfs_f)
# coord_c = SpatialCoordinate(vfs_c)

eddy_turnover_time = 2.5
spde_spin_up_time = eddy_turnover_time * 20


def read_in_chkpoint(pv, name, filename):
    with DumbCheckpoint(utility.output_directory(filename), mode=FILE_READ) as chk:
        chk.load(pv, name)


def load_chk_point(filename, func, n="Vorticity"):
    with DumbCheckpoint(filename, mode=FILE_READ) as chk:
        chk.load(func, name=n)


def project_fine_to_coarse(pv_fine, coarse_msh, coarse_dg_space):
    """
     project u_fine (finer resolution) to a given coarse mesh
     this is done by directly projecting the fine res streamfunction to the coarse res

    :param pv_fine: fine res function to be projected down
    :param coarse_msh: coarse mesh
    :param coarse_dg_space: coarse DG space
    """
    # Obtain the fine streamfunction
    pv_fine_msh = pv_fine.function_space().mesh()
    pv_fine_solver = euler.EulerSolver(euler.EulerParams(1., 0.1, pv_fine_msh))
    pv_fine_solver.psi_solve_given_q(pv_fine)

    cfs_psi = FunctionSpace(coarse_msh, "CG", 1)
    psi0 = Function(cfs_psi)
    inject(pv_fine_solver.psi0, psi0)

    # solve for q weakly given the projected stream function
    q0 = Function(coarse_dg_space, name="Vorticity")
    euler.EulerSolver.solve_for_q_given_psi(psi0, q0)

    # v = Function(VectorFunctionSpace(coarse_msh, "CG", 1), name="Velocity")
    # v.project(euler.EulerSolver(euler.EulerParams(1., 0.1, coarse_msh)).gradperp(psi0))
    # output_file = File(utility.output_directory('coarse_grain.pvd', '/TestResults'))
    # output_file.write(q0, v, t=0)
    return q0


def coarse_average_fine(pv_fine, coarse_msh, coarse_dg_space, _k_sqr):
    """
    coarse graining of a fine resolution pv function using the helmholtz operator

    :param pv_fine:
    :param coarse_msh:
    :param coarse_dg_space:
    :param _k_sqr:
    :return:
    """
    pv_fine_msh = pv_fine.function_space().mesh()
    pv_fine_solver = euler.EulerSolver(euler.EulerParams(1., 0.1, pv_fine_msh))
    pv_fine_solver.psi_solve_given_q(pv_fine)
    pv_fine_solver.psi0.project(pv_fine_solver.helmholtz_solver(pv_fine_solver.psi0, _k_sqr))

    cfs_psi = FunctionSpace(coarse_msh, "CG", 1)
    psi0 = Function(cfs_psi)
    inject(pv_fine_solver.psi0, psi0)

    # solve for q weakly given the projected stream function
    q0 = Function(coarse_dg_space, name="Vorticity")
    euler.EulerSolver.solve_for_q_given_psi(psi0, q0)

    # v = Function(VectorFunctionSpace(coarse_msh, "CG", 1), name="Velocity")
    # v.project(euler.EulerSolver(euler.EulerParams(1., 0.1, coarse_msh)).gradperp(psi0))
    # output_file = File(utility.output_directory('coarse_grain_helmholtz.pvd', '/TestResults'))
    # output_file.write(q0, v, t=0)
    return q0


def output_f_at_grid(f, result, n0=64):
    x_array = np.array([1. * _x / n0 for _x in range(n0 + 1)])
    for i in x_array:
        for j in x_array:
            grid_ij = f.at(np.array([i, j]), tolerance=1e-10)
            result.append(grid_ij)


def l2_error(f1, f2):
    """
    assume f1 and f2 are on the same mesh structure, compute the l2 norm of their difference
    :param f1:
    :param f2:
    :return:
    """
    return norm(assemble(f1, f2))


def get_coarse_mesh(res):
    """
    given resoltuion, return coarse mesh
    :param res:
    :return:
    """
    dictionary = {8: mesh_hierarchy[0], 16: mesh_hierarchy[1], 32: mesh_hierarchy[2], 64: mesh_hierarchy[3], 128: mesh_hierarchy[4],
                  256: mesh_hierarchy[5], 512: mesh_hierarchy[6]}

    return dictionary[res]


def get_coarse_dt(res):
    """
    given resolution, return coarse dt

    :param res: resolution
    :return:
    """
    dictionary = {64: 0.02, 128: 0.01, 256: 0.005, 512: 0.0025}

    return dictionary[res]
