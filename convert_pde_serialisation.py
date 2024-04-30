import sys
from firedrake import *
from utility import Workspace
from tqg.solver import TQGSolver
from firedrake_utility import TorusMeshHierarchy
# from firedrake import pi, ensemble,COMM_WORLD, PETSc

from tqg.example2 import TQGExampleTwo as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm

root = 0

def fname(wspace, file_index):
    return wspace.output_name("_pde_data_{}".format(file_index), "data") 

def rpsi_file_name(file_index, workspace, increment_id=None):
    _fname_= "_regularised_psi_increment_{}".format(file_index)
    _fname = _fname_ + "_{}".format(increment_id) if increment_id !=None else _fname_ 
    return workspace.output_name(_fname, "data") 

if __name__ == "__main__":
    flag_perturb = True 
    for arg in sys.argv[1:]:
        print(type(arg), arg)
        flag_perturb = bool(int(arg))
        print(flag_perturb)
    
    if 1:
        nx = 512
        T = 0.001
        dt = 0.0002
        dump_freq = 5
        solve_flag = False
        write_visual = False 
        procno = ensemble_comm.rank 

        mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()

        # project DG to CG for .at
        gFScg = FunctionSpace(mesh, "CG", 1)
        gPVcg = Function(gFScg)
        gBcg  = Function(gFScg)
        gPSIcg= Function(gFScg)
        
        params = Example2params(T, dt, mesh, bc='x', alpha=None)
        tqg_solver = TQGSolver(params)

        input_workspace = Workspace("/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, 'none'))
        #output_workspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")
        output_workspace = Workspace("/home/wpan1/Data/PythonProjects/TQGExample2/res_512_alpha_none")

        myprint = PETSc.Sys.Print
        
        #ndays = len(range(0, 4276))
        index_range_for_perturbation = range(0, 2775)
        ndays_range = range(2775, 4276)

        file_index = 0
        _sub_folder = "PDESolution/Perturbation" if flag_perturb else "PDESolution"
        ndays_range = index_range_for_perturbation if flag_perturb else ndays_range

        data_chk = None

        if spatial_comm.rank == 0:
            data_chk = DumbCheckpoint(output_workspace.output_name('pde_data', _sub_folder), single_file=False, mode=FILE_CREATE, comm=ensemble_comm)
        
        coords_shape = None
        coords = None
        
        pv_converted_cg = None
        b_converted_cg  = None
        pv_converted_dg = None
        b_converted_dg  = None
        psi_converted_cg = None

        if spatial_comm.rank == root:
            mesh_serialised = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=ensemble_comm).get_fine_mesh()
            vfs = VectorFunctionSpace(mesh_serialised, "CG", 1)
            fscg = FunctionSpace(mesh_serialised, "CG", 1)
            fsdg = FunctionSpace(mesh_serialised, "DG", 1)

            pv_converted_cg = Function(fscg, name="pv_converted_cg").assign(0)
            b_converted_cg  = Function(fscg, name="b_converted_cg").assign(0)
            pv_converted_dg = Function(fsdg, name="PotentialVorticity")
            b_converted_dg  = Function(fsdg, name="Buoyancy")
            psi_converted_cg= Function(fscg, name="Streamfunction").assign(0)

            coords = Function(vfs).interpolate(SpatialCoordinate(mesh_serialised))
            coords_shape = coords.dat.data.shape

        coords_shape = spatial_comm.bcast(coords_shape, root=root)
        coords_all = np.zeros(coords_shape)

        if spatial_comm.rank == root:
            assert coords_all.shape == coords_shape, "coords_all and coords have different shapes"
            coords_all += coords.dat.data

        spatial_comm.Bcast(coords_all, root=root)


        for file_index in ndays_range:
        #if 1:
            myprint(file_index, ' convert and save', fname(input_workspace, file_index), ', perturb' if flag_perturb else 'no perturb')

            tqg_solver.load_initial_conditions_from_file(fname(input_workspace, file_index))

            #if 0 and ensemble_comm.rank == 0:
            #    output_file = File(output_workspace.output_name('test_original.pvd', _sub_folder), comm=spatial_comm)
            #    #print(output_workspace.output_name('test.pvd'))
            #    output_file.write(tqg_solver.initial_cond, tqg_solver.initial_b, tqg_solver.psi0, time=0)
        #if 1:

            gPVcg.project(tqg_solver.initial_cond)
            gBcg.project( tqg_solver.initial_b)
            gPSIcg.assign( tqg_solver.psi0)

            pv_cg_at_values = np.asarray(gPVcg.at(coords_all, tolerance=1e-10))
            b_cg_at_values  = np.asarray(gBcg.at( coords_all, tolerance=1e-10))         
            psi_cg_at_values= np.asarray(gPSIcg.at(coords_all, tolerance=1e-10))

            if spatial_comm.rank == root:
                assert pv_cg_at_values.shape == pv_converted_cg.dat.data.shape, "pv_cg shape inconsistent" 
                assert b_cg_at_values.shape  == b_converted_cg.dat.data.shape, "b_cg shape inconsistent" 
                assert psi_cg_at_values.shape== psi_converted_cg.dat.data.shape, "psi_cg shape inconsistent"

                pv_converted_cg.assign(0)
                b_converted_cg.assign(0)
                psi_converted_cg.assign(0)

                pv_converted_cg.dat.data[:] += pv_cg_at_values
                b_converted_cg.dat.data[:]  += b_cg_at_values
                psi_converted_cg.dat.data[:]+= psi_cg_at_values

                pv_converted_dg.project(pv_converted_cg)
                b_converted_dg.project(b_converted_cg)

                #output_file = File(output_workspace.output_name('test.pvd'))
                #output_file.write(pv_converted_cg, pv_converted_dg, b_converted_cg, b_converted_dg, time=0)
                #print('done convert ', pv_converted_cg.dat.data.shape)

            #print(spatial_comm.rank, type(pv_converted_cg))

            if spatial_comm.rank == root:
                #output_file = File(output_workspace.output_name('test_{}.pvd'.format(file_index), _sub_folder), comm=ensemble_comm)
                #print(output_workspace.output_name('test.pvd'))
                #output_file.write( pv_converted_dg,  b_converted_dg, psi_converted_cg, time=file_index)
                #with open(output_workspace.output_name('test.npy'), 'wb') as f:
                #    np.save(f, pv_converted_dg.dat.data)
                #    np.save(f, pv_converted_cg.dat.data)
                #data_chk = DumbCheckpoint(output_workspace.output_name('test', _sub_folder), single_file=True, mode=FILE_CREATE, comm=ensemble_comm)
                data_chk.store(pv_converted_dg, name="PotentialVorticity")
                data_chk.store(b_converted_dg, name="Buoyancy")
                data_chk.store(psi_converted_cg,name="Streamfunction")
                data_chk.new_file()

        if spatial_comm.rank == 0:
            data_chk.close()

                #print('done saving npy')
            


