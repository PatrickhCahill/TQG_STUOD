#   Copyright   2020
#
#   wei_example.py
#
#   Horizontal Rayleigh Benard convection example 2
#

from tqg.initial_conditions_abstract_class import *


class TQGExampleTwoTest(TQGParams):

    def set_initial_conditions(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_q.interpolate( -exp(-5.0*(2*pi*x[1] - pi)**2) )
        # self.initial_q.interpolate( -exp(-5.0*(x[1] - pi)**2) )
        self.bathymetry.interpolate( (cos(2.0*pi*x[0]) + 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*x[0])/3.0))
        self.rotation.assign(0)
        self.initial_b.interpolate(sin(2*pi*(x[0]-0.25)))
        # self.initial_b.interpolate( -1.0 / (1.0 + exp(-(2*pi*x[0]- pi))) + 0.5 )
        # self.initial_b.interpolate( -1.0 / (1.0 + exp(-(x[0]- pi))) + 0.5 )

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation

    def set_forcing(self):
        x = SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())
        #return Function(self.TQG_fem_params.Vdg).interpolate(10.* sin(16.0*pi*x[1]))
        return Function(self.TQG_fem_params.Vdg).assign(0.01 * self.initial_b)

    def set_damping_rate(self):
        return Constant(0.00001)



class TQGExampleTwo(TQGParams):

    def set_initial_conditions(self, random_bathymetry=False):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_q.interpolate( -exp(-5.0*(2*pi*x[1] - pi)**2) )
        # self.initial_q.interpolate( -exp(-5.0*(x[1] - pi)**2) )
        self.bathymetry.interpolate( cos(2*pi*x[0]) + 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*x[0])/3.0)
        
        if random_bathymetry:        
            import numpy as np
            indices = np.random.choice(self.bathymetry.dat.data.shape[0], int(self.bathymetry.dat.data.shape[0]*0.25), replace=False)
            gscaling = np.random.normal(1., 0.55, indices.shape)
            ## scale a quater of the values
            # print("bathymetry shape " , self.bathymetry.dat.data.shape)
            # print(np.max(gscaling), np.min(gscaling))
            # i want to randomly pick a subset to scale, not the whole thing
            self.bathymetry.dat.data[indices] *= gscaling

        self.rotation.assign(0)
        self.initial_b.interpolate(0.1* sin(2*pi*x[0]))
        # self.initial_b.interpolate( -1.0 / (1.0 + exp(-(2*pi*x[0]- pi))) + 0.5 )
        # self.initial_b.interpolate( -1.0 / (1.0 + exp(-(x[0]- pi))) + 0.5 )

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation


class TQGExampleTwoForML(TQGParams):

    def set_initial_conditions(self):
        """
        Initial condition taken from paper.
        This initial condition is to be spun up to some equilibrium initial state, from which
        subsequent simulation continues
        """
        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_q.interpolate( -exp(-5.0*(2*pi*x[1] - pi)**2) )
        # self.initial_q.interpolate( -exp(-5.0*(x[1] - pi)**2) )
        self.bathymetry.interpolate(0.1*(cos(2*pi*x[0]) + 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*x[0])/3.0))
        self.rotation.assign(0)
        self.initial_b.interpolate(sin(2*pi*x[0]))
        #self.initial_b.interpolate(sin(8*pi*x[0])*cos(4*pi*(x[1]+x[0])))
        #self.initial_b.interpolate(sin(16*pi*x[0])*cos(16*pi*x[1]))
        #self.initial_b.interpolate((sin(12*pi*x[0])*cos(2*pi*x[1])+sin(4*pi*x[0])*cos(4*pi*x[1])))

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation
