#   Copyright   2020
#
#   wei_example.py
#
#   THIS CASE SHOULD BE STATIONARY
#

from tqg.initial_conditions_abstract_class import *


class TQGExampleThreeL(TQGParams):

    def set_initial_conditions(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_q.interpolate( cos(2.0*pi*x[1]))
        # self.bathymetry.interpolate( cos(x[0])  + 0.5 * cos(2.0 * x[0]) + cos( 3.0*x[0] )/3.0 )
        # self.bathymetry.assign(5.)
        self.bathymetry.interpolate( sin(2.0*pi*x[1]))
        # self.bathymetry.interpolate( 1. - (2.0 * pi* x[0]/10.)**2 / 2. + (2.0*pi*x[0]/10.)**4 / 24. - (2.0*pi*x[0]/10.)**6 / 720. )
        self.rotation.assign(0)
        # self.initial_b.interpolate( sin( (x[0] - shift)) )
        self.initial_b.interpolate( sin(2.0*pi*x[1])-1)

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation
