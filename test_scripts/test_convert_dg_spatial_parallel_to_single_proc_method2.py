# test convert a spatially parallelised firedrake function
# to a non-parallelised function
# using .at coordinate method


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
    myprint('errnorm gref gtest: ', errornorm(gref, gtest)/norm(gtest))

    func = None
    coords_shape = None
    coords = None
    ltest = None

    if spatial_comm.rank == root:
        # non-parallelised
        mesh2 = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=ensemble_comm).get_fine_mesh()
        vfs = VectorFunctionSpace(mesh2, "DG", 1)
        fs = FunctionSpace(mesh2, "DG", 1)
        func = Function(fs)
        ltest = Function(fs)
        lref = Function(fs)
        lcoords = SpatialCoordinate(mesh2)

        coords = Function(vfs).interpolate(lcoords)

        func.assign(0)
        ltest.interpolate(20*sin(2*pi*lcoords[0])*cos(2*pi*lcoords[1]))
        lref = Function(fs).interpolate(20*(sin(8*pi*lcoords[0]) + sin(4*pi*lcoords[0])*sin(2*pi*lcoords[1])))

        coords_shape = coords.dat.data.shape
        print("coords_shape ", coords_shape)

    coords_shape = spatial_comm.bcast(coords_shape, root=root)
    coords_all = np.zeros(coords_shape)

    if spatial_comm.rank == root:
        #print(type(coords.dat.data), coords.dat.data.shape)
        coords_all += coords.dat.data

    spatial_comm.Bcast(coords_all, root=root)
    gref_ = np.asarray(gref.at(coords_all, tolerance=1e-10))

    myprint("gref_.shape ", gref_.shape)
    
    if spatial_comm.rank == root:
        assert gref_.shape == func.dat.data.shape
        func.dat.data[:] += gref_

        print('proc ', root, '- ', 'func: ', norm(func), ' ltest: ', norm(ltest), ' lref: ', norm(lref))
        print('proc ', root, ' errornorm lref, ltest: ', errornorm(lref, ltest)/norm(ltest))
        print('proc ', root, ' errornorm func, ltest: ', errornorm(func, ltest)/norm(ltest))


