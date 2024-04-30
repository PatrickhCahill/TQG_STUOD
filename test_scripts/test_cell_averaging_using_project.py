from relative_import_header import *
from firedrake import *
from utility import Workspace
from firedrake_utility import TorusMeshHierarchy
from tqg.solver import TQGSolver
#import numpy as np
from random import randint, sample


def pde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("pde_data_{}".format(file_index), sub_dir) 

def spde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("ensemble_member_{}_fs".format(file_index), sub_dir) 


if __name__ == "__main__":
    nx = 512 
    cnx = 128

    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y").get_fine_mesh()
    cmesh = TorusMeshHierarchy(cnx, cnx, 1., 1., 0, period="y").get_fine_mesh()
    
    coordinate = Function(VectorFunctionSpace(cmesh, "CG", 1)).interpolate(SpatialCoordinate(cmesh))

    fs_cg = FunctionSpace(mesh, "CG", 1)
    fs_dg = FunctionSpace(mesh, "DG", 1)

    cfs_cg = FunctionSpace(cmesh, "CG", 1)
    
    #fs_cg_0 = FunctionSpace(mesh, "CG", 0)
    fs_dg_0 = FunctionSpace(mesh, "DG", 0)

    #buoyancy_cg = Function(fs_cg)
    buoyancy_dg = Function(fs_dg, name="Buoyancy")

    cbuoyanc_cg = Function(cfs_cg, name="Buoyancy_bar")

    wspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

    h5_data_name = pde_data_fname(wspace, 0, 'PDESolution')

    #h5_data_name = spde_data_fname(wspace, 48, 'EnsembleMembers')

    with DumbCheckpoint(h5_data_name, mode=FILE_READ) as chk:
        chk.load(buoyancy_dg, name="Buoyancy")

    print(norm(buoyancy_dg))
    buoyancy_bar = project(buoyancy_dg, fs_dg_0)
    cbuoyanc_cg.dat.data[:] += buoyancy_bar.at(coordinate.dat.data, tolerance=1e-10)

    #print(norm(ssh_dg), norm(ssh))
    foutput_file = File(wspace.output_name("buoyancy_project_average.pvd","TestFiles"))
    foutput_file.write(buoyancy_dg)
    foutput_file = File(wspace.output_name("cbuoyanc_project_average.pvd","TestFiles"))
    foutput_file.write(cbuoyanc_cg)
