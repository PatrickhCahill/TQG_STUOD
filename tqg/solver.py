#
# Author Wei Pan
#
# TQG solver
#

from firedrake import *
import numpy as np
from stqg.solver import STQGSolver


class TQGSolver(STQGSolver):

    def __init__(self, tqg_params):
        STQGSolver.__init__(self, tqg_params)
        self.solver_name = 'TQG solver'

    def perturb_bathymetry(self, grid_percentage=0.25, stddev=0.55):
        import numpy as np
        # set seed so we can replicate the result
        np.random.seed(250)

        indices = np.random.choice(self.bathymetry.dat.data.shape[0], int(self.bathymetry.dat.data.shape[0]*grid_percentage), replace=False)
        gscaling = np.random.normal(1., stddev, indices.shape)
        
        self.bathymetry.dat.data[indices] *= gscaling
        np.random.seed(None)

    def perturb_bathymetry_deterministic(self, freq):

        x = SpatialCoordinate(self.Vcg.mesh())

        perturb = Function(self.Vcg).interpolate( cos(freq*pi*x[0]) / freq/2. )

        self.bathymetry.dat.data[:] += perturb.dat.data[:]
