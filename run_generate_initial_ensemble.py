import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace
from stqg.generate_initial_ensemble import STQGEnsembleGenerator, STQGEnsembleGeneratorCoarse
from firedrake import pi, ensemble,COMM_WORLD
#from firedrake import *
from firedrake.petsc import PETSc
from tqg.example2 import TQGExampleTwo as Example2params

desired_spatial_rank = 1
ensemble = ensemble.Ensemble(COMM_WORLD, desired_spatial_rank)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":
    nx = 512
    cnx = 32  
    
    #T = 0.001 # was 0.05
    dt = 0.00025
    cdt = 0.001
    dump_freq = 4
    dump_freq_c = 1
    solve_flag = False
    write_visual = False 
    
    batch_id = None 
    ref_file_id = 0 # id of pde solution for reference
    switch_off_bathymetry = False

    if len(sys.argv) > 1:
        batch_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        ref_file_id = int(sys.argv[2])
    if len(sys.argv) > 3:
        switch_off_bathymetry = bool(sys.argv[3])

    #print(ref_file_id, flush=True)

    if ensemble_comm.rank ==0:
        print("batch_id: ", batch_id, ", ensemble_member_id: ",  ensemble_comm.rank + batch_id * ensemble_comm.size, " ref_file_id: ", ref_file_id, flush=True)

    ensemble_size = ensemble_comm.size 

    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=spatial_comm).get_fine_mesh()
    cmesh = TorusMeshHierarchy(cnx, cnx, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh() 
    angry_dolphin_params_c = Example2params(0.001, cdt, cmesh, bc='x', alpha=None)
    
    # It doesn't matter what the actual params are, we just need a param object to pass to the Solver constructor
    angry_dolphin_params = Example2params(0.001, dt, mesh, bc='x', alpha=None)
    
    if 1:
        ensemble_generator = STQGEnsembleGenerator(angry_dolphin_params, ensemble_size, ensemble, batch_id=batch_id, truth_ref_file_id=ref_file_id, switch_off_bathymetry=switch_off_bathymetry)
        
        import numpy as np
        ensemble_generator.generate_an_ensemble_member_Dan(cnx, cnx, ensemble, dump_freq, epsilon=0.1) #, epsilon=np.abs(np.random.normal(0,3) ))
    else: 
        if ref_file_id == 0:
            #print("fine res gen", flush=True)
            ensemble_generator = STQGEnsembleGenerator(angry_dolphin_params, ensemble_size, ensemble, batch_id=batch_id, truth_ref_file_id=ref_file_id, switch_off_bathymetry=switch_off_bathymetry)
            ensemble_generator.generate_an_ensemble_member(cnx, cnx, ensemble, dump_freq)
        if 0:
            ensemble_generator_c = STQGEnsembleGeneratorCoarse(angry_dolphin_params_c, angry_dolphin_params, ensemble_size, ensemble, batch_id=batch_id, truth_ref_file_id=ref_file_id-1, switch_off_bathymetry=switch_off_bathymetry)
            #else:
            #print("coarse res gen", flush=True)
            #print("run generate_an_ensemble_member")
            xi_scaling=1.
            ensemble_generator_c.generate_an_ensemble_member(ensemble, dump_freq_c, xi_scaling)
    

