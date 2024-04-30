from firedrake import *
import numpy as np

nx = 64
L = 1
mesh = PeriodicRectangleMesh(nx,nx,L,L, direction='x')

Vcg = FunctionSpace(mesh, "CG", 1)
Vdg = FunctionSpace(mesh, "DG", 1)

bathymetry = Function(Vcg)
buoyancy = Function(Vcg)
bathymetry_dg = Function(Vdg)

x = SpatialCoordinate(mesh)

# bathymetry.interpolate( cos(2.0*pi*x[1]) + cos(4.0*pi*x[1]) )
# bathymetry_dg.interpolate( cos(2.0*pi*x[1]) + cos(4.0*pi*x[1]) )
bathymetry.assign(0)
bathymetry_dg.assign(0)
buoyancy.interpolate(   sin(2.0*pi*x[1]) )

gradperp = gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))

buoyancy_dg = Function(Vdg)
buoyancy_dg.interpolate(sin(2.0*pi*x[1]))

print("try cg bathymetry")
print("buoyancy (vcg)", assemble(-0.5*dot(gradperp(bathymetry), grad(buoyancy)) * dx))
print("buoyancy (vdg)", assemble(-0.5*dot(gradperp(bathymetry), grad(buoyancy_dg)) * dx))

print("buoyancy div (vcg)", assemble(-0.5*div(gradperp(bathymetry)*buoyancy) * dx))
print("buoyancy div (vdg)", assemble(-0.5*div(gradperp(bathymetry)*buoyancy_dg) * dx))

print("try dg bathymetry")
print("buoyancy (vcg)", assemble(-0.5*dot(gradperp(bathymetry_dg), grad(buoyancy)) * dx))
print("buoyancy (vdg)", assemble(-0.5*dot(gradperp(bathymetry_dg), grad(buoyancy_dg)) * dx))

print("buoyancy div (vcg)", assemble(-0.5*div(gradperp(bathymetry_dg)*buoyancy) * dx))
print("buoyancy div (vdg)", assemble(-0.5*div(gradperp(bathymetry_dg)*buoyancy_dg) * dx))
