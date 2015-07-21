from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import numpy


parser = ArgumentParser("Replace simulation halo mass with Nbody simulation halo mass",
        description=
        """
############################################################################
#  ##    ## ########   #######  ########  ##    ## ##    ## #### ########  # 
#  ###   ## ##     ## ##     ## ##     ##  ##  ##  ##   ##   ##     ##     # 
#  ####  ## ##     ## ##     ## ##     ##   ####   ##  ##    ##     ##     # 
#  ## ## ## ########  ##     ## ##     ##    ##    #####     ##     ##     # 
#  ##  #### ##     ## ##     ## ##     ##    ##    ##  ##    ##     ##     # 
#  ##   ### ##     ## ##     ## ##     ##    ##    ##   ##   ##     ##     # 
#  ##    ## ########   #######  ########     ##    ##    ## ####    ##     # 
############################################################################
        Replace the friends of friend halo mass produced by nbodykit of a simulation (e.g. fPM), 
        with Nbody simulation halo mass(e.g. RunPB), according to their haloID's.
        """,
        epilog=
        """
        This script is written by Yu Feng, as part of `nbodykit'. 
        """
        , formatter_class=RawTextHelpFormatter)

parser.add_argument("sim_halo", 
        help='Name of the halo file, only Nbodykit format is supported in this script.')
parser.add_argument("Nbody_halo", 
        help='Name of the halo file, only Nbodykit format is supported in this script.')
parser.add_argument("output", 
        help='Basename of the output, a file with $filename.halo.rpbmass would be produced.')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

import nbodykit
from nbodykit import files

def main():

    sim = files.HaloFile(ns.sim_halo)
    nbody = files.HaloFile(ns.Nbody_halo)

    sim_mass = sim.read('Mass')
    nbody_mass = nbody.read('Mass')
    sim_pos = sim.read('Position')
    sim_vel = sim.read('Velocity')
    sim_link_l = sim.linking_length
    sim_nhalo = sim.nhalo
    nbody_nhalo = nbody.nhalo
    minlength = min(nbody_nhalo,sim_nhalo)

    print "starts to replace masses of halos."
    with open(ns.output + '.halo.rpbmass', 'w') as ff:
        numpy.int32(minlength).tofile(ff)
        numpy.float32(sim_link_l).tofile(ff)
        numpy.int32(nbody_mass[:minlength]).tofile(ff)
        numpy.float32(sim_pos[:minlength]).tofile(ff)
        numpy.float32(sim_vel[:minlength]).tofile(ff)

main()
