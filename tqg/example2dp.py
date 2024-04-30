#   Copyright   2020
#
#   wei_example.py
#
#   Test case for Laura
#

from tqg.initial_conditions_abstract_class import *


class TQGExampleTwoDoublyPeriodic(TQGParams):

    def set_initial_conditions(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        # self.initial_q.interpolate( -exp(-5.0*(x[1] - pi)**2) )
        # self.initial_q.interpolate( -0.0001 * (4.0*pi**2 + 1) * cos(2.0 * pi * x[1]) )
        #self.initial_q.interpolate( cos(2.0 * pi * x[1]) )
        self.initial_q.interpolate( cos(2.0 * pi * x[0]) )
        #self.bathymetry.interpolate( cos(2*pi*x[0]) ) # + 0.5 * cos(4.0*pi * x[0]) ) # + cos(6.0*pi*x[0])/3.0)
        self.bathymetry.interpolate( cos(2.0 *pi*x[1]) )
        # self.bathymetry.assign(0)
        self.rotation.assign(0)
        # self.initial_b.interpolate( cos(2.0 * pi * x[0]) ) # sin(2.0 * pi * x[0]) )
        self.initial_b.interpolate( cos(2.0 *pi *x[1]))




        return self.initial_q, self.initial_b, self.bathymetry, self.rotation
