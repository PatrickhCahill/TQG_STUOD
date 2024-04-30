from firedrake import *

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "DG", 1)
test_func = Function(V, name="dump")

with DumbCheckpoint("./dump", single_file=False, mode=FILE_CREATE) as chk:
#     chk.store(test_func)
    for i in range(3):
        chk.new_file()
        chk.store(test_func)



