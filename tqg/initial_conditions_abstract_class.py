#
#   Author: Wei Pan
#   Copyright   2020
#
#   tqg.py
#
#   Defines OOP classes for the tqg model and corresponding solver
#

from abc import ABC, abstractmethod
from firedrake import *

class TQGFemParams(object):
    """ firedrake finite element parameters """

    def __init__(self, dt, msh, cg_deg=1):

        self.Vdg = FunctionSpace(msh, "DG", 1)
        self.Vcg = FunctionSpace(msh, "CG", cg_deg)
        self.Vu = VectorFunctionSpace(msh, "DG", 1)
        self.x = SpatialCoordinate(msh)
        self.dt = dt
        self.facet_normal = FacetNormal(msh)


class TQGModelParams(object):
    """ TQG model parameters """

    def __init__(self, fem_params, t, bc='y'):
        # ## slip boundary condition
        self.time_length = t
        if (bc == 'y') or ( bc == 'x'):
            self.bcs = DirichletBC(fem_params.Vcg, 0.0, 'on_boundary')
            self.dgbcs = DirichletBC(fem_params.Vdg, 0.0, 'on_boundary')
        else:
            self.bcs = []
            self.dgbcs =[]
        # self.bcs = DirichletBC(fem_params.Vcg, 0., (1, 2))
        # self.bcs = []


class TQGParams(ABC):
    """ Wrapper class for TQGModelParams and TQGFemParams """

    def __init__(self, T, dt, msh, bc='y', cg_deg=1, alpha=64):
        self.TQG_fem_params = TQGFemParams(dt, msh, cg_deg)
        self.TQG_model_params = TQGModelParams(self.TQG_fem_params, T, bc)
        
        self.initial_q = Function(self.TQG_fem_params.Vdg, name="PotentialVorticity")
        self.initial_b = Function(self.TQG_fem_params.Vdg, name="Buoyancy")
        self.bathymetry = Function(self.TQG_fem_params.Vcg, name="Bathymetry")
        self.rotation = Function(self.TQG_fem_params.Vdg, name="Rotation")
        self.alphasqr = Constant(1./alpha**2) if alpha != None else Constant(0)
        super().__init__()

    @abstractmethod
    def set_initial_conditions(self):
        pass

    def set_forcing(self):
        return Function(self.TQG_fem_params.Vdg).assign(0) 

    def set_damping_rate(self):
        return Constant(0.)
