from firedrake import *
import numpy as np

N = 256 

mesh = UnitSquareMesh(N,N)

def test(deg):
    Vcg = FunctionSpace(mesh, "CG", deg)
    
    x = SpatialCoordinate(mesh)

    # func = Function(Vcg).interpolate(sin(x[0])*sin(x[1]))
    func = Function(Vcg).interpolate(x[0]**3 + x[1]**3)
    g = Function(Vcg).interpolate(x[0]+x[1])
    g2 = Function(Vcg).interpolate(6.*x[0] + 6.*x[1])

    # print( assemble( inner(grad(func), grad(div(grad(func))))*dx) )
    # print( assemble( inner(grad(g), grad(g)) *dx))
    # print( norm(grad(g)) )
    using_g =  assemble( inner(grad(g), grad(div(grad(func)))) * dx) 
    using_as_vec =  assemble( inner( as_vector([1., 1.]), grad(div(grad(func)))) * dx) 
    ans = -4.*np.sin(1)*(1. - np.cos(1))
    ans = 12.
    print("deg: ", deg, ", ")
    print("using_g: ", using_g,  np.abs(ans - using_g))
    print("using_as_vec", using_as_vec, np.abs(ans - using_as_vec))
    print(np.abs(assemble( inner(  grad(div(grad(func))), grad(div(grad(func))) )*dx))) # ,    "\n")
    print(np.abs(assemble( inner( grad(g2), grad(div(grad(func))))*dx)))
    print( assemble( inner( as_vector([6., 6.]), grad(div(grad(func)))) * dx) , "\n")


for i in range(1,5):
    test(i)
