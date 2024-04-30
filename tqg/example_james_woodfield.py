from tqg.initial_conditions_abstract_class import *


class TQGExampleJames(TQGParams):
    """stationary """

    def set_initial_conditions(self):
    
        print("here", flush=True)


        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_q.interpolate( sin(2*pi*x[0]) *sin(2*pi*x[1]) )
        self.initial_b.interpolate( sin(2*pi*x[0]) *sin(2*pi*x[1]) )

        self.bathymetry.interpolate(0.005 * sin(2*pi*x[0]) * sin(2*pi*x[1]) )
        self.rotation.interpolate(   0.02 * sin(2*pi*x[0]) * sin(2*pi*x[1]) )

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation



class TQGExampleJames2(TQGParams):
    """ non-stationary """

    def set_initial_conditions(self):
    
        print("here", flush=True)


        x= SpatialCoordinate(self.TQG_fem_params.Vdg.mesh())

        self.initial_q.interpolate( sin(2*pi*x[0]) *sin(2*pi*x[1]) )
        self.initial_b.interpolate( sin(2*pi*x[0]) *sin(2*pi*x[1]) )

        self.bathymetry.interpolate(0.005 * cos(2*pi*x[0]) * cos(2*pi*x[1]) )
        self.rotation.interpolate(   0.02 * sin(2*pi*x[0]) * sin(2*pi*x[1]) )

        return self.initial_q, self.initial_b, self.bathymetry, self.rotation



