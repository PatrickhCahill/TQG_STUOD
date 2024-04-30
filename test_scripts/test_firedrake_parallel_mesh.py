import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utility import Workspace, commandline_parser
from firedrake import * # pi, ensemble,COMM_WORLD, PETSc
import numpy as np
from mpi4py import MPI

nprocs_spatial = COMM_WORLD.size
ensemble = Ensemble(COMM_WORLD, 1 ) # COMM_WORLD.size)

spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm
global_comm = ensemble.global_comm
root = 0

myprint = PETSc.Sys.Print
#myprint("spatial_comm: ", spatial_comm.rank, spatial_comm.size)
#myprint("ensemble_comm: ", ensemble_comm.rank, ensemble_comm.size)
#myprint("global_comm: ", global_comm.rank, global_comm.size)

res = 256 
res_c = 128 

# mesh = UnitSquareMesh(res, res, comm=spatial_comm)
mesh = PeriodicSquareMesh(res, res, 1.0, direction='x', quadrilateral=True, comm=spatial_comm)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V).interpolate(sin(2*pi*x)*cos(2*pi*y))
norm_of_u = norm(u)

#mesh3 = PeriodicSquareMesh(res_c, res_c, 1.0, direction='x', quadrilateral=True) #, comm=ensemble_comm)
#x__, y__ = SpatialCoordinate(mesh3)
#V2 = FunctionSpace(mesh3, "CG", 1)
#u_ = Function(V2).interpolate(sin(2*pi*x__)*cos(2*pi*y__))
#
#norm_of_u_ = norm(u_)
#
#print(norm_of_u_, abs(norm_of_u - norm_of_u_))

#recvbuf = None #spatial_comm.gather(u.dat.data.tolist(), root=0)
#sendbuf = u.dat.data
#
#sendcounts = np.array(spatial_comm.gather(len(sendbuf), root))
#if spatial_comm.rank == root:
#    recvbuf = np.empty(sum(sendcounts), dtype=np.double)

print("ens proc: {}".format(ensemble_comm.rank), norm_of_u, "{}".format(ensemble_comm.size))

if 0:

    workspace = Workspace("/home/wpan1/Data/PythonProjects/TestMPIFiredrake")

    if ensemble_comm.rank == 0:
        output_file = File(workspace.output_name("u.pvd"))
        output_file.write(u)

    #spatial_comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)

    mesh2 = None
    u2 = None
    u2_ = None
    output_file2 = File(workspace.output_name("u2.pvd"))

    coords_shape = None
    coords = None

    if spatial_comm.rank == root:
        #mesh2 = UnitSquareMesh(res, res, comm=ensemble_comm)
        mesh2 = PeriodicSquareMesh(res_c, res_c, 1.0, direction='x', quadrilateral=True, comm=ensemble_comm)
        vfs = VectorFunctionSpace(mesh2, "CG", 1)
        fs = FunctionSpace(mesh2, "CG", 1)
        coords = Function(vfs).interpolate(SpatialCoordinate(vfs))
        u2 = Function(fs)
        # coords_all += coords.dat.data
        coords_shape = coords.dat.data.shape
        x_, y_ = SpatialCoordinate(mesh2)
        u2_ = Function(fs).interpolate(sin(2*pi*x_)*cos(2*pi*y_))

    coords_shape = spatial_comm.bcast(coords_shape, root=0)
    coords_all = np.zeros(coords_shape)

    if spatial_comm.rank == root:
        coords_all += coords.dat.data

    spatial_comm.Bcast(coords_all, root=root)
    mydat = np.asarray(u.at(coords_all, tolerance=1e-10))

    if spatial_comm.rank == root:
        u2.dat.data[:] += mydat
        output_file2.write(u2)
        u2_h5_name = workspace.output_name("u2")
        _norm_u2 = norm(u2)
        _norm_u2_ = norm(u2_)
        print(abs(_norm_u2 - norm_of_u), abs(_norm_u2_ - norm_of_u), abs(_norm_u2 - _norm_u2_))
        np.save(workspace.output_name("u2_dat"), mydat)

    print("proc {}".format(spatial_comm.rank), coords_shape)






#if spatial_comm.rank == 0 and 0:
#    mesh2 = UnitSquareMesh(res, res, comm=ensemble_comm)
#    fs = FunctionSpace(mesh2, "CG", 1)
#    u2 = Function(fs)
#    
#    print(type(recvbuf), len(recvbuf))
#
#    print(u2.dat.data.shape, recvbuf.shape, norm(u2))
#    print(recvbuf)
#    print(u2.dat.data)
#    u2.dat.data[:] += recvbuf
#    print(u2.dat.data)
#    print(norm(u2))
#
#    print(u2.dat.data.shape)
#    output_file2.write(u2)
#
#if ensemble_comm.rank == 0 and 0:
#    output_file2 = File(workspace.output_name("u2.pvd"))
#    output_file2.write(u2)

