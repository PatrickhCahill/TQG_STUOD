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

    def __init__(self, dt, msh):

        self.Vdg = FunctionSpace(msh, "DG", 1)
        self.Vcg = FunctionSpace(msh, "CG", 1)
        self.Vu = VectorFunctionSpace(msh, "DG", 1)
        self.x = SpatialCoordinate(msh)
        self.dt = dt
        self.facet_normal = FacetNormal(msh)


class TQGModelParams(object):
    """ TQG model parameters """

    def __init__(self, fem_params, t, bc='y'):
        # ## slip boundary condition
        self.time_length = t
        # self.r = Constant(0)
        if bc == 'y':
            self.bcs = DirichletBC(fem_params.Vcg, 0.0, 'on_boundary')
        else:
            self.bcs = []
        # self.bcs = DirichletBC(fem_params.Vcg, 0., (1, 2))
        # self.bcs = []
        # self.dgbcs = DirichletBC(fem_params.Vdg, 0.0, 'on_boundary')


class TQGParams(ABC):
    """ Wrapper class for TQGModelParams and TQGFemParams """

    def __init__(self, T, dt, msh, bc='y'):
        self.TQG_fem_params = TQGFemParams(dt, msh)
        self.TQG_model_params = TQGModelParams(self.TQG_fem_params, T, bc)
        self.initial_q = Function(self.TQG_fem_params.Vdg)
        self.initial_b = Function(self.TQG_fem_params.Vdg)
        self.bathymetry = Function(self.TQG_fem_params.Vcg)
        self.rotation = Function(self.TQG_fem_params.Vdg)
        super().__init__()

    @abstractmethod
    def set_initial_conditions(self):
        pass
