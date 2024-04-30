#! /usr/bin/env python
#
from fenics import *
from mshr import *

def heat_implicit ( ):

#*****************************************************************************80
#
## heat_implicit solves the 2D heat equation on rectangle with interior hole.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 October 2018
#
#  Author:
#
#    John Burkardt
#
  import matplotlib.pyplot as plt
#
#  Define the domain.
#
  circle_x = 0.5
  circle_y = 0.5
  circle_r = 0.25

  domain = Rectangle(Point(-1.0,-1.0), Point(1.,1.)) \
        - Circle(Point(circle_x,circle_y),circle_r)
#
#  Mesh the domain.
#
  mesh = generate_mesh ( domain, 10 )
#
#  Plot the mesh.
#
  plot ( mesh, title = 'heat_implicit Mesh' )
  filename = 'heat_implicit_mesh.png'
  plt.savefig ( filename )
  print ( '  Graphics saved as "%s"' % ( filename ) )
  plt.close ( )
#
#  Define the function space.
#
  V = FunctionSpace ( mesh, "Lagrange", 1 )
#
#  Define the boundary conditions.
#  These could depend on time as well as space.
#
  rect_u = 10.0
  def rect_on ( x, on_boundary ):
    return ( on_boundary )

  rect_bc = DirichletBC ( V, rect_u, rect_on )

  circle_u = 100.0
  def circle_on ( x, on_boundary ):
    return ( on_boundary )

  circle_bc = DirichletBC ( V, circle_u, circle_on )

  bc = [ rect_bc, circle_bc ]
#
#  Define the trial functions (u) and test functions (v).
#
  u = TrialFunction ( V )
  v = TestFunction ( V )
#
#  The diffusivity is a constant.
#
  k = Constant ( 1.0 )
#
#  The source term is zero. 
#
  f = Constant ( 0.0 )
#
#  Define time things.
#
  t_init = 0.0
  t_final = 5.0
  t_num = 20
  dt = ( t_final - t_init ) / t_num
#
#  Create U_INIT.
#
  u_init = Expression ( "40.0", degree = 10 )

  u_old = interpolate ( u_init, V )

  fvt = ( u_old + dt * f ) * v * dx
  Auvt = u * v * dx + dt * dot ( grad(u), grad(v) ) * dx
#
#  U <-- the initial condition.
#
  u = Function ( V )
#
#  T <-- the initial time.
#
  t = t_init
#
#  Time loop.
#
  for j in range ( 0, t_num + 1 ):

    if ( j % 2 == 0 ):
      label = 'Time = %g' % ( t )
      plot ( u_old, title = label )
      filename = 'heat_implicit_solution_%d.png' % ( j )
      plt.savefig ( filename )
      print ( '  Graphics saved as "%s"' % ( filename ) )
      plt.close ( )
#
#  T <-- T + DT
#
    t = t + dt
#
#  Solve Auvt = fvt for U.
#
    solve ( Auvt == fvt, u, bc )
#
#  Copy U_OLD values from U
#
    u_old.assign ( u )
#
#  Terminate.
#
  return

def heat_implicit_test ( ):

#*****************************************************************************80
#
## heat_implicit_test tests heat_implicit.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 October 2018
#
#  Author:
#
#    John Burkardt
#
  import time

  print ( time.ctime ( time.time() ) )
#
#  Report level = only warnings or higher.
#
  level = 30
  set_log_level ( level )

  print ( '' )
  print ( 'heat_implicit_test:' )
  print ( '  FENICS/Python version' )
  print ( '  Time-dependent heat equation.' )
  print ( '  Implicit solver.' )

  heat_implicit ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'heat_implicit_test:' )
  print ( '  Normal end of execution.' )
  print ( '' )
  print ( time.ctime ( time.time() ) )
  return

if ( __name__ == '__main__' ):

  heat_implicit_test ( )
