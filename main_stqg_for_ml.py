#   Copyright 2020
#
#   STUOD TQG Horizontal Rayleigh Benard convection
#   
#
import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
from stqg.solver import STQGSolver
from firedrake import pi, ensemble,COMM_WORLD, File
from firedrake.petsc import PETSc 
from tqg.example2 import TQGExampleTwoForML as Example2params
import numpy as np

spatial_size = 1
ensemble = ensemble.Ensemble(COMM_WORLD, spatial_size)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm

def get_grid_values(nx, ny, func):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    xv, yv = np.meshgrid(x, y, indexing='ij')
    func_grid = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            func_grid[i,j] += np.asarray(func.at([xv[i,j], yv[i,j]], tolerance=1e-10))
    return func_grid


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = opts['time']
    nx = opts['resolution']
    dt = opts['time_step']
    dump_freq = opts['dump_freq']
    alpha = opts['alpha']
    alpha = None
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']
    do_save_spectrum = None

    PETSc.Sys.Print("time horizon: {}, res: {}, dt: {}, dump_req: {}, alpha: {}".format(T, nx, dt, dump_freq, alpha), flush=True)

    workspace = Workspace("/home/wpan1/Development/Data/STQGForML")
    
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=spatial_comm).get_fine_mesh()

    test_files = "TestFiles/"
    test_files = "TestFiles_test/"
    #test_files = "TestFiles_requested_format/"

        
    params = Example2params(T,dt,mesh, bc='x', alpha=alpha)
    # params = Example3lparams(T,dt,TorusMeshHierarchy(nx,nx,1.,1.,0,"both",comm=comm).get_fine_mesh(), bc='',alpha=alpha)
    solver = STQGSolver(params) 
    #solver.load_initial_conditions_from_file(workspace.output_name("ensemble_member_0", "EnsembleMembers"), comm=spatial_comm)

    if 0:
        zeta_fname = workspace.output_name("zetas.npy", test_files + "ParamFiles")
        visual_output_name = workspace.output_name("spde_visuals", test_files + "visuals")
        data_output_name   = workspace.output_name("spde_data", test_files + "data")

    if 0:
        noise = solver.solve(dump_freq, visual_output_name, data_output_name, ensemble_comm, do_save_data=do_save_data, do_save_visual=do_save_visual, do_save_spectrum=do_save_spectrum, res=nx, zetas_file_name=zeta_fname )

        import numpy as np
        np.save(workspace.output_name("noise", test_files + "data"), noise)
        print(noise.shape)

    if 1:
        nx = 65
        #load noise file
        noise = np.load(workspace.output_name("noise.npy", test_files + "data"))
        print(noise.shape)

    #if 1:
        from firedrake import FunctionSpace, VectorFunctionSpace, SpatialCoordinate, Function, sin, cos, pi
        import numpy as np
        # generate one zeta for STQG
        vcg = FunctionSpace(mesh, "CG", 1)
        x = SpatialCoordinate(mesh)
        # original is 0.0002 and 0.0002 for scaling, with dt = 0.01
        zeta1 = Function(vcg).interpolate(0.0002*(sin(12*pi*x[0])*cos(2*pi*x[1])+sin(4*pi*x[0])*cos(4*pi*x[1]))/4)
        zeta2 = Function(vcg).interpolate(0.0002*sin(8*pi*x[0])*cos(4*pi*(x[0]+x[1])))
        
        zeta1_grid = get_grid_values(nx, nx, zeta1)
        zeta2_grid = get_grid_values(nx, nx, zeta2)

        #print(zeta1_grid.shape)
        #print(zeta2_grid.shape)

        coloured_noise = np.zeros((nx, nx, 10001, 2))
        #print(coloured_noise.shape)

        for i in range(1, 10001, 1):
            coloured_noise[:,:,i,0] += zeta1_grid * noise[2*(i-1)]
            coloured_noise[:,:,i,1] += zeta2_grid * noise[2*(i-1)+1]

        np.save(workspace.output_name("coloured_noise", test_files + "ParamFiles"), coloured_noise)
        
        if 0:
            _zeta = np.asarray([zeta1.dat.data, zeta2.dat.data])
            print(_zeta.shape)
            np.save(workspace.output_name("zetas", test_files + "ParamFiles"), _zeta)
            coords = (Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(x)).dat.data
            np.save(workspace.output_name("coords", test_files + "ParamFiles"), coords)

    if 1:
        # save space_time_grid
        space_time_grid = np.zeros((nx, nx, 10001, 3))
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, nx)

        xv, yv = np.meshgrid(x, y, indexing='ij')
        for t in range(10001):
            print(t, dt*t)
            for i in range(nx):
                for j in range(nx):
                    space_time_grid[i,j,t,:] = [xv[i,j], yv[i,j], dt*t]
        np.save(workspace.output_name("space_time_grid", test_files + "ParamFiles"), space_time_grid)


    if 0:
        nx = 65

        #save bathymetry
        #np.save(workspace.output_name("bathymetry", test_files + "ParamFiles"), params.bathymetry.dat.data)
        np.save(workspace.output_name("bathymetry", test_files + "ParamFiles"), get_grid_values(nx, nx, params.bathymetry))

        from firedrake import FunctionSpace, VectorFunctionSpace, SpatialCoordinate, Function, sin, cos, pi
        temp_cg_func = Function(FunctionSpace(mesh, "CG", 1))
        
        space_time_dim = (nx, nx, 10001)
        b_grid_values = np.zeros(space_time_dim)
        q_grid_values = np.zeros(space_time_dim) 
        psi_grid_values = np.zeros(space_time_dim)

        for index in range(10001):
            print(index)
            solver.load_initial_conditions_from_file(workspace.output_name("spde_data_{}".format(index), test_files + "data"), comm=spatial_comm)
            # buoyancy grid values
            temp_cg_func.project(solver.initial_b)
            #b_grid_values.append(temp_cg_func.dat.data)
            b_grid_values[:,:,index] += get_grid_values(nx, nx, temp_cg_func)
            
            # pv grid values
            temp_cg_func.project(solver.initial_cond)
            #q_grid_values.append(temp_cg_func.dat.data)
            q_grid_values[:,:,index] += get_grid_values(nx, nx, temp_cg_func)

            # psi grid values
            #psi_grid_values.append(solver.psi0.dat.data)
            psi_grid_values[:,:,index] += get_grid_values(nx, nx, solver.psi0)

        #b_grid_values = np.asarray(b_grid_values)
        #q_grid_values = np.asarray(q_grid_values)
        #psi_grid_values=np.asarray(psi_grid_values)

        print(b_grid_values.shape)
        print(q_grid_values.shape)
        print(psi_grid_values.shape)

        np.save(workspace.output_name("buoyancy", test_files + "data"), b_grid_values)
        np.save(workspace.output_name("pv", test_files + "data"), q_grid_values)
        np.save(workspace.output_name("psi", test_files + "data"), psi_grid_values)

