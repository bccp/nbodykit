from mpi4py_test import MPITester
import sys
import os.path

tester = MPITester(os.path.abspath(__file__), "nbodykit")

tester.main(sys.argv[1:])
