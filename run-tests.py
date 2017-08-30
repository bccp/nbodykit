import sys; sys.path.pop(0)
from runtests.mpi import Tester
import os.path

tester = Tester(os.path.abspath(__file__), "nbodykit")
tester.main(sys.argv[1:])
