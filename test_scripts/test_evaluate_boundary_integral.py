from firedrake import *

nx = 512 

mesh = PeriodicRectangleMesh(nx,nx,1.,1., direction="y")

fs = FunctionSpace(mesh, "CG", 1)
fsdg = FunctionSpace(mesh, "DG",1 )
x  = SpatialCoordinate(mesh)

func_fine_res  = Function(fs).interpolate(sin(2*pi*x[0])*sin(2*pi*x[1]))
pv = Function(fs).interpolate(-8*pi*pi*sin(2*pi*x[0])*sin(2*pi*x[1]))
q_dg = Function(fsdg)


q_cg_space = fs
q = TrialFunction(q_cg_space)
v = TestFunction(q_cg_space)

bc = DirichletBC(q_cg_space, pv, 'on_boundary')

a = func_fine_res * v * dx + (dot(grad(v), grad(func_fine_res))) * dx
L = q * v * dx

q0 = Function(q_cg_space)
solve(L == a, q0, bcs=[bc])
q_dg.project(q0)

print(norm(pv), norm(q0))
#facet_normal = FacetNormal(mesh)
#
#flag = True
#
#k_sqr = 1.
#fs = func_fine_res.function_space()
#cg_fs = FunctionSpace(fs.mesh(), 'CG', 1)
#f = Function(cg_fs).project(func_fine_res)
#u = TrialFunction(cg_fs)
#v = TestFunction(cg_fs)
#c_sqr = Constant(1.0 / k_sqr)
#
#a = (c_sqr * dot(grad(v), grad(u)) + v * u) * dx
#L = f * v * dx
##g = assemble(dot(grad(f), facet_normal))
##print(type(g))
#
##a_flux = c_sqr * v * g * ds
##L_ = L + a_flux
#
#bc = DirichletBC(cg_fs, f, 'on_boundary') if flag == True else []
##print(bc)
#
#u = Function(cg_fs)
#solve(a == L, u, bcs=bc, solver_parameters={'ksp_type': 'cg'})

#print(norm(u))

# --- b equation -----
#un = 0.5 * (dot(self.gradperp(self.psi0), fem_params.facet_normal) \
#    + abs(dot(self.gradperp(self.psi0), fem_params.facet_normal)))
#
#_un_ = dot(self.gradperp(self.psi0), fem_params.facet_normal)
#_abs_un_ = abs(_un_)
#
#b = TrialFunction(Vdg)
#p = TestFunction(Vdg)
#a_mass_b = p * b * dx
#a_int_b = (dot(grad(p), -self.gradperp(self.psi0) * b)) * dx
#
## a_flux_b = (p('+') * _un_('+') + p('-') * _un_('-') ) * avg(b) * dS  # central flux
## a_flux_b = (dot(jump(p), un('+')*b('+') - un('-')*b('-')) )*dS  # upwinding
#a_flux_b =  0.5*jump(p)*(2*_un_('+')*avg(b) + _abs_un_('+')*jump(b))*dS  
#
#arhs_b = a_mass_b - dt * (a_int_b + a_flux_b)
#
#b_problem = LinearVariationalProblem(a_mass_b, action(arhs_b, self.b1), self.db1 \
