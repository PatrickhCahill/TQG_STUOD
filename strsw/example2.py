#   Copyright   2020
#
#   wei_example.py
#
#   Horizontal Rayleigh Benard convection example 2
#

from tqg.initial_conditions_abstract_class import *


class TQGExampleTwo(TQGParams):

    def set_initial_conditions(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_q.interpolate( -exp(-5.0*(2*pi*x[1] - pi)**2) )
        # self.initial_q.interpolate( -exp(-5.0*(x[1] - pi)**2) )
        self.bathymetry.interpolate( cos(2*pi*x[0]) + 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*x[0])/3.0)
        self.rotation.assign(0)
        self.initial_b.interpolate(sin(2*pi*x[0]))
        # self.initial_b.interpolate( -1.0 / (1.0 + exp(-(2*pi*x[0]- pi))) + 0.5 )
        # self.initial_b.interpolate( -1.0 / (1.0 + exp(-(x[0]- pi))) + 0.5 )

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation
