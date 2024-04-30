from mpi4py import MPI

comm = MPI.COMM_WORLD
procno = comm.Get_rank()
nprocs = comm.Get_size()
print(procno, "/", nprocs)


if (procno == 0):
    print("proc write", procno)
    with open("./test_{}.txt".format(procno), 'w') as f:
        f.write("this is a test, from procno {}\n".format(procno))

