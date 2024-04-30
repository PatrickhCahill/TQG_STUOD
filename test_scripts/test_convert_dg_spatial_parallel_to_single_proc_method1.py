# test convert a spatially parallelised firedrake function
# to a non-parallelised function
# using gather dat.data directly (doens't seem to work)


from relative_import_header import *
from firedrake import *
from firedrake_utility import TorusMeshHierarchy

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm

root = 0

if __name__ == "__main__":
    nx = 512
    procno = ensemble_comm.rank 
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
    gfs = FunctionSpace(mesh, "DG", 1)
    gtest = Function(gfs)
    gcoords = SpatialCoordinate(mesh)

    myprint = PETSc.Sys.Print

    # parallelised
    gtest.interpolate(20*sin(2*pi*gcoords[0])*cos(2*pi*gcoords[1]))
    gref = Function(gfs).interpolate(20*(sin(8*pi*gcoords[0]) + sin(4*pi*gcoords[0])*sin(2*pi*gcoords[1])))

    myprint('norm gref: ', norm(gref), ' gtest: ',  norm(gtest))
    myprint('errnorm gref gtest: ', errornorm(gref, gtest)/norm(gref))

    func = None
    coords_shape = None
    coords = None

    recvbuf = None
    ltest = None

    if spatial_comm.rank == root:
        # non-parallelised
        mesh2 = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=ensemble_comm).get_fine_mesh()
        fs = FunctionSpace(mesh2, "DG", 1)
        func = Function(fs)
        ltest = Function(fs)
        lref = Function(fs)
        lcoords = SpatialCoordinate(mesh2)
        func.assign(0)
        ltest.interpolate(20*sin(2*pi*lcoords[0])*cos(2*pi*lcoords[1]))
        lref = Function(fs).interpolate(20*(sin(8*pi*lcoords[0]) + sin(4*pi*lcoords[0])*sin(2*pi*lcoords[1])))
        # coords_shape = coords.dat.data.shape

        recvbuf = np.zeros(func.dat.data.shape)

        #print(func.dat.data.shape)
    #sendcounts = np.array(comm.gather(len(sendbuf), root))
    #coords_shape = spatial_comm.bcast(coords_shape, root=root)
    #coords_all = np.zeros(coords_shape)
    _temp_ = gref.dat.data[:] #tqg_solver.initial_cond.dat.data
    # _data = spatial_comm.Gather(_temp_, root=root)
    spatial_comm.Gather(sendbuf=_temp_, recvbuf=recvbuf, root=root)

    if spatial_comm.rank == root:
        # coords_all += coords.dat.data
        # print(len(func.dat.data), len(recvbuf))
        # print(type(recvbuf))
        func.dat.data[:] += recvbuf
        # print(_data)
        print('proc ', root, '- ', 'func: ', norm(func), ' ltest: ', norm(ltest), ' lref: ', norm(lref))
        print('proc ', root, ' errornorm lref, ltest: ', errornorm(lref, ltest)/norm(lref))
        print('proc ', root, ' errornorm func, ltest: ', errornorm(func, ltest)/norm(func))

