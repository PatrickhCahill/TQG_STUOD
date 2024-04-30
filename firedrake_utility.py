"""firedrake utility

This module defines common variables and functions for firedrake functions
"""
from firedrake import MeshHierarchy, TorusMesh, PeriodicSquareMesh, PeriodicRectangleMesh, RectangleMesh, COMM_WORLD


class TorusMeshHierarchy(object):
    """Generates a doubly periodic rectangular mesh, a.k.a. a torus.
    """

    def __init__(self, nR, nr, R, r, refinement_level, period="y", quad=False, comm=COMM_WORLD):
        """Torus mesh __init__ method.

        Defines a mesh hierarchy structure for a torus mesh.

        Args:
            nR (int): The coarsest number of cells in the major direction (min 3).
            nr (int): The coarsest number of cells in the minor direction (min 3).
            R (int): The major radius.
            r (int): The minor radius.
            refinement_level (int): Mesh hierarchy level (min 0). Level 0 returns base mesh.
        """
        if nR < 3 or nr < 3:
            raise ValueError("Must have at least 3 cells in each direction.")
        
        if refinement_level < 0:
            raise ValueError("Refinement level must not be less than 0.")

        self._refinement_level = refinement_level
        self._mesh_hierarchy = {refinement_level: PeriodicRectangleMesh(nR,nr,R,r, direction=period,comm=comm, quadrilateral=quad)} 
        
    def get_fine_mesh(self):
        """This method returns the finest mesh in the mesh hierarchy.
        
        Returns:
            Mesh.
        """
        return self._mesh_hierarchy[self._refinement_level]

    def get_coarse_mesh(self):
        """This method returns the coarsest mesh in the mesh hierarchy.
        
        Returns:
            Mesh.
        """
        pass
        # return self._mesh_hierarchy[0]
