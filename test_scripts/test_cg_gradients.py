from firedrake import *

msh = UnitSquareMesh(64,64,quadrilateral=False)

gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0)))

CG1 = FunctionSpace(msh, "CG",1 )
DG1 = FunctionSpace(msh, "DG",1)
VDG1 = VectorFunctionSpace(msh, "DG",1)

vregpsi = Function(VDG1, name="vrpsi")
vpsi = Function(VDG1, name="vpsi")
fcg1  = Function(CG1,name="psi")
regpsi = Function(CG1, name="rpsi")
g = Function(DG1,name="q")

x,y = SpatialCoordinate(msh)

g.interpolate(sin(2*pi*x)*sin(2*pi*y))

# ----- elliptic --------
psi = TrialFunction(CG1)
phi = TestFunction(CG1)

Apsi = (dot(grad(phi), grad(psi)) + phi*psi)*dx
Lpsi = -g*phi*dx
bc = DirichletBC(CG1, 0.0, 'on_boundary')

psi_problem = LinearVariationalProblem(Apsi, Lpsi, fcg1, bc)
psi_solver = LinearVariationalSolver(psi_problem, solver_parameters={'ksp_type':'cg', 'pc_type':'sor'})
psi_solver.solve()

# ----- helmholtz -------
alphasqr = Constant(1./32**2)
rpsi = TrialFunction(CG1)
rphi = TestFunction(CG1)
Arpsi = (alphasqr*dot(grad(rpsi),grad(rphi)) + rphi*rpsi)*dx
Lrpsi = fcg1 * rphi*dx
rpsi_problem = LinearVariationalProblem(Arpsi, Lrpsi,regpsi, bc) 
rpsi_solver = LinearVariationalSolver(rpsi_problem, solver_parameters={'ksp_type':'cg','pc_type':'sor'})
rpsi_solver.solve()

vregpsi.project(gradperp(regpsi))
vpsi.project(-gradperp(fcg1))
workspace = "/home/wpan1/Data/PythonProjects/test_cg_gradients"

fout = File(workspace + "/out.pvd")
fout.write(g, fcg1,regpsi, vregpsi, vpsi)
