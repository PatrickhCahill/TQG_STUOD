from firedrake import *

nx = 64
L = 1.0
mesh = PeriodicUnitSquareMesh(nx, nx,direction="x", quadrilateral=True)
x = SpatialCoordinate(mesh)

vcg = FunctionSpace(mesh, "CG", 1)
vu = VectorFunctionSpace(mesh, "DG", 1)
vdg = FunctionSpace(mesh, "DG", 1)

my_const = Constant(as_vector([1., 0.]))
velocity = Function(vu).interpolate(my_const)
q = Function(vdg)

bathymetry = Function(vcg, name="Bathymetry")
#bathymetry.interpolate( (cos(4.0*pi * x[0]) + cos(6.0*pi*x[0]))*sin(8.0*pi*x[1]) )
#gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))
#q.project(dot(gradperp(bathymetry), velocity))
# print(norm(q))
 
#bathymetry.interpolate( (cos(2.0*pi*x[0])+ 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*x[0])/3.0)*sin(2.0*pi*x[1]) )
bathymetry.interpolate( exp(-10.*((x[0] - 0.5)**2+ x[1]**2) ) )
bathymetry.interpolate( exp(-10.*((x[0] -0.5 )**2+ (x[1]-1.)**2) ) )
bathymetry.interpolate( tanh(4*(x[1]-0.5)) )
#output_file = File("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_none/visuals/bathymetry.pvd".format(nx))
output_file = File("/home/wpan1/Development/Data/PythonProjects/bathymetry.pvd")
output_file.write(bathymetry)
