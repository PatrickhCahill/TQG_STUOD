from firedrake import *
workspace = "/home/wpan1/Data/PythonProjects/heat_implicit/"

nx = 128
mesh = UnitSquareMesh(nx, nx)

V = FunctionSpace(mesh, "CG", 1)

u0 = Function(V)
u_old = Function(V, name="solution")

Dt = 0.01
dt = Constant(Dt)
T = 1.
dump = 1
coeff = Constant(0.05)

u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)

u_old.interpolate(sin(2*pi*x)*sin(2*pi*y))


output = File(workspace + "heat_implicit.pvd")
output.write(u_old, time=0)

a = ( v*u + dt * coeff * dot(grad(v), grad(u)) ) * dx
L = u_old * v * dx

bc = DirichletBC(V, 0., 'on_boundary')

t = 0
while(t < (T-Dt/2)):
    
    solve(a == L, u0, bcs=bc, solver_parameters={'ksp_type': 'cg'})
    u_old.assign(u0)
    
    t += Dt
    print(round(t,2))
    output.write(u_old, time=round(t,2))
