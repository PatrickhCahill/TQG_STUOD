from tqg.initial_conditions_abstract_class import *


class TQGExampleZero(TQGParams):

    def set_initial_conditions(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_b.interpolate( sin(2*pi*x[0])*sin(2*pi*x[1]))
        self.bathymetry.interpolate( cos(2*pi*x[0]) + 0.5 * cos(4.0 *pi*x[0]) + cos(6.0*pi*x[0])/3.0 )
        self.rotation.assign(0)
        self.initial_q.interpolate( sin(4.0*pi*x[0])*sin(4.0*pi*x[1]) +
                                    0.4*cos(3.0*pi*x[0])*cos(3.0*pi*x[1])+
                                    0.3*cos(5.0*pi*x[0])*cos(2.0*pi*x[1])) 

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation
