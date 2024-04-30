# import tqg
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace
from tqg.example0 import TQGExampleZero as Example0params
# from tqg.example1 import TQGExampleOne as Example1params
# from tqg.example1b import TQGExampleOneB as Example1bparams
# from tqg.example2 import TQGExampleTwo as Example2params
# from tqg.example2b import TQGExampleTwoB as Example2bparams
# from tqg.example2dp import TQGExampleTwoDoublyPeriodic as Example2dpparams
from tqg.example3 import TQGExampleThree as Example3params
from tqg.example3l import TQGExampleThreeL as Example3lparams
# from tqg.example3b import TQGExampleThreeB as Example3bparams
# from tqg.example3c import TQGExampleThreeC as Example3cparams
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
# from euler import EulerSolver, EulerParams
from firedrake import pi , COMM_WORLD

import sys

if __name__ == "__main__":

    workspace = None

    _arg = str(sys.argv[1])
    T = float(sys.argv[2])
    n0 = int(sys.argv[3])
    dt = float(sys.argv[4])
    dump_freq = int(sys.argv[5])

    vblimiter = False
        
    if (_arg == '3'):
        L = 1.
        nx = 64
        dt = 0.0025 
        dump_freq = 40
        print("ex3 ", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_64")
        solver = TQGSolver(Example3params(T, dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc=''))
    if (_arg == '3_parallel'):
        L = 1.
        nx =n0
        dt = 0.0001
        dump_freq = 1
        if (COMM_WORLD.rank == 0):
            print("ex 3 parallel", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_parallel")
        solver = TQGSolver(Example3params(T,dt, TorusMeshHierarchy(nx, nx, L,L,0,"both").get_fine_mesh(), bc=''))

    elif (_arg == '3_smaller_dt'):
        L =1.
        nx = 64
        dt = 0.001
        dump_freq = 100
        print("ex 3", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_64_smaller_dt")
        solver = TQGSolver(Example3params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc=''))
    
    elif (_arg == '0_smaller_dt'):
        L = 1.
        nx = n0
        print("ex 0 smaller dt", nx, T, dt, L, nx)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example0_smaller_dt")
        solver = TQGSolver(Example0params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(),bc=''))
    elif (_arg == '0_256_smaller_dt_quadrilateral'):
        L = 1.
        nx = n0 
        print("ex 0 smaller dt", nx, T, dt, L, nx, flush=True)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example0_{}_smaller_dt_quadrilateral".format(nx))
        solver = TQGSolver(Example0params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both",True).get_fine_mesh(),bc=''))

   
    elif (_arg == '0_256_smaller_dt'):
        L = 1.
        nx = n0 
        print("ex 0 smaller dt", nx, T, dt, L, nx, flush=True)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example0_{}_smaller_dt".format(nx))
        solver = TQGSolver(Example0params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(),bc=''))

    elif (_arg == '0_256_smaller_dt_quadrilateral_regularised'):
        L = 1.
        nx = n0 
        print("ex 0 smaller dt quad reg", nx, T, dt, L, dump_freq, flush=True)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example0_{}_smaller_dt_quadrilateral_regularised".format(nx))
        solver = RTQGSolver(Example0params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both",True).get_fine_mesh(),bc='',res=0.5))
   
    elif (_arg == '0_256_smaller_dt_regularised_smallT'):
        L = 1.
        nx = 512 
        T=50
        dump_freq = 40
        dt = 0.00025
        print("ex 0 smaller dt reg", nx, T, dt, L, dump_freq, flush=True)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example0_{}_smaller_dt_regularised_smallT".format(nx))
        solver = RTQGSolver(Example0params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(),bc='',res=0.15))

    elif (_arg == '0_256_smaller_dt_regularised'):
        L = 1.
        nx = n0 
        print("ex 0 smaller dt reg", nx, T, dt, L, dump_freq, flush=True)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example0_{}_smaller_dt_regularised".format(nx))
        solver = RTQGSolver(Example0params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(),bc='',res=0.5))

    elif(_arg == '0'):
        L = 1.
        nx = n0 
        print("ex 0", nx, T, dt, L, nx)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example0")
        solver = TQGSolver(Example0params(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(),bc=''))

    elif (_arg == '3_l_smaller_dt_regularised'):
        L =1.
        nx = 64
        dt = 0.001
        dump_freq = 1000 
        print("ex 3 regularised", nx, T, dt, L, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_l_64_smaller_dt_regularised")
        solver = RTQGSolver(Example3lparams(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc='',res=0.5))

    elif (_arg == '3_l_smaller_dt_quadrilateral_regularised'):
        L =1.
        nx = 64
        dt = 0.001
        dump_freq = 1000 
        print("ex 3 quadrilateral regularised", nx, T, dt, L, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_l_64_smaller_dt_quadrilateral_regularised")
        solver = RTQGSolver(Example3lparams(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both",True).get_fine_mesh(), bc='',res=0.5))

    elif (_arg == '3_l_smaller_dt'):
        L =1.
        nx = 64
        dt = 0.0001
        dump_freq = 1000
        print("ex 3", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_l_64_smaller_dt")
        solver = TQGSolver(Example3lparams(T,dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc=''))

    elif (_arg == '3_limiter'):
        vblimiter =True
        L = 1.
        nx = 64
        dt = 0.0025 
        dump_freq = 40
        print("ex3 ", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_64_limiter")
        solver = TQGSolver(Example3params(T, dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc=''))
        
    elif (_arg == '3_128' ):
        L = 1.
        nx = 128
        dt = 0.0025
        dump_freq = 40
        print("ex3 ", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_128")
        solver = TQGSolver(Example3params(T, dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc=''))
        
    elif (_arg == '3_128_limiter'):
        vblimiter = True
        L = 1.
        nx = 128
        dt = 0.0025
        dump_freq = 40
        print("ex3 ", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_128_limiter")
        solver = TQGSolver(Example3params(T, dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc=''))
    elif (_arg == '3_128_smaller_dt' ):
        L = 1.
        nx = 128
        dt = 0.001 
        dump_freq = 100
        print("ex3 ", nx, T, dt, L, nx, 1)
        workspace = Workspace("/home/wpan1/Data/PythonProjects/tqg_example3_128_smaller_dt")
        solver = TQGSolver(Example3params(T, dt, TorusMeshHierarchy(nx,nx,L,L,0,"both").get_fine_mesh(), bc=''))
 
    else:
        raise ValueError

    solver.solve(dump_freq, workspace.output_name("tqg_test", "visuals"), \
            workspace.output_name("tqg_test", "plots"), vblimiter)

