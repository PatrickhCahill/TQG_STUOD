
#   Copyright   2021
#

from tqg.initial_conditions_abstract_class import *


class TQGTheoryExampleTwo(TQGParams):

    def set_initial_conditions(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x = SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())
        # x_cg = SpatialCoordinate(self.TQG_fem_params.Vcg.mesh())

        self.initial_q.interpolate(sin(8.0*pi*x[0])*sin(8.0*pi*x[1]) +
                                      0.4*cos(6.0*pi*x[0])*cos(6.0*pi*x[1])+
                                      0.3*cos(10.0*pi*x[0])*cos(4.0*pi*x[1]) +
                                      0.02*sin(2.0*pi*x[1]) + 0.02*sin(2.0*pi*x[0])  ) #)

        self.bathymetry.interpolate( (cos(4.0*pi * x[0]) + cos(6.0*pi*x[0]))*sin(8.0*pi*x[1]) )
        # self.bathymetry.interpolate( (cos(2.0*pi*x[0])+ 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*x[0])/3.0)*sin(2.0*pi*x[1]) )
        # self.rotation.interpolate(sin(2.0*pi*x[1]))
        self.rotation.assign(0)
        self.initial_b.interpolate( sin(2.0*pi*x[1]) - 1 )

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation
                                      
