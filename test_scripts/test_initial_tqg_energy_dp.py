from firedrake import *

nx = 64

mesh = PeriodicUnitSquareMesh(nx, nx, direction="both")
vdg = FunctionSpace(mesh, "DG", 1)
vcg = FunctionSpace(mesh, "CG", 1)

x = SpatialCoordinate(mesh)

q = Function(vdg).interpolate(cos(2.*pi * x[0]))
b = Function(vdg).interpolate(cos(2.*pi*x[1]))
psi = Function(vcg)
h = Function(vcg).interpolate(cos(2.*pi*x[1]))

print(assemble(-0.5*h*b*dx))
