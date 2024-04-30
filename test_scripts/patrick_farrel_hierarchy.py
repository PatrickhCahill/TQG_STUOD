from firedrake import VectorFunctionSpace, Function, Constant, Mesh, par_loop, CylinderMesh, MeshHierarchy
from firedrake import READ, WRITE, dx, distribution_parameters, HierarchyBase
import numpy


class Problem(object):
    def __init__(self, baseN, nref):
        super().__init__()
        self.baseN = baseN
        self.nref = nref
    @staticmethod
    def periodise(m):
        coord_fs = VectorFunctionSpace(m, "DG", 1, dim=2)
        old_coordinates = m.coordinates
        new_coordinates = Function(coord_fs)
        # make x-periodic mesh
        # unravel x coordinates like in periodic interval
        # set y coordinates to z coordinates
        periodic_kernel = """double Y,pi;
            Y = 0.0;
            for(int i=0; i<old_coords.dofs; i++) {
                Y += old_coords[i][1];
            }
            pi=3.141592653589793;
            for(int i=0;i<new_coords.dofs;i++){
            new_coords[i][0] = atan2(old_coords[i][1],old_coords[i][0])/pi/2;
            if(new_coords[i][0]<0.) new_coords[i][0] += 1;
            if(new_coords[i][0]==0 && Y<0.) new_coords[i][0] = 1.0;
            new_coords[i][0] *= Lx[0];
            new_coords[i][1] = old_coords[i][2]*Ly[0];
            }"""
        cLx = Constant(1)
        cLy = Constant(1)
        par_loop(periodic_kernel, dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "Lx": (cLx, READ),
              "Ly": (cLy, READ)})
        return Mesh(new_coordinates)
    @staticmethod
    def snap(mesh, N, L=1):
        coords = mesh.coordinates.dat.data
        coords[...] = numpy.round((N/L)*coords)*(L/N)
    def mesh(self):
        base = CylinderMesh(self.baseN, self.baseN, 1.0, 1.0, longitudinal_direction="z",
                     distribution_parameters=distribution_parameters)
        mh = MeshHierarchy(base, self.nref, distribution_parameters=distribution_parameters)
        meshes = tuple(self.periodise(m) for m in mh)
        mh = HierarchyBase(meshes, mh.coarse_to_fine_cells, mh.fine_to_coarse_cells)
        for (i, m) in enumerate(mh):
            if i > 0:
                self.snap(m, self.baseN * 2**i)
        for mesh in mh:
            mesh.coordinates.dat.data[...] = 2*mesh.coordinates.dat.data - 1
        return mh[-1]