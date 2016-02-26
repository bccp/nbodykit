from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import numpy


parser = ArgumentParser("Matching particle ID's in different halos in halo catalogues",
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
        It can be used as identifying the merging halos in simulations by picking two redshifts of halo catelogue, 
        as well as comparing different halo catalogues in different simulations.
        """,
        epilog=
        """
        This script is written by Yu Feng, as part of `nbodykit'. 
        """
        , formatter_class=RawTextHelpFormatter)

parser.add_argument("snapfilename1", 
        help='basename of the snapshot1, only runpb format is supported in this script')
parser.add_argument("snapfilename2", 
        help='basename of the snapshot2, only runpb format is supported in this script')
parser.add_argument("halolabel1", 
        help='basename of the halo label files1, only nbodykit format is supported in this script')
parser.add_argument("halolabel2", 
        help='basename of the halo label files2, only nbodykit format is supported in this script')
parser.add_argument("halocatalogue1", 
        help='halocatalogue1')
parser.add_argument("halocatalogue2", 
        help='halocatalogue2')
parser.add_argument("output", 
        help='Basename of the output, print all the corresponding halo ID of partiles in file1 as well as file2, a file with $filename.halo.rpbmass would be produced.')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

import nbodykit
from nbodykit import files
from nbodykit import halos
from nbodykit.corrfrompower import corrfrompower
from kdcount import correlate

def main(): 


    SNAP1, LABEL1 = None, None
    SNAP1 = files.Snapshot(ns.snapfilename1, files.TPMSnapshotFile)
    LABEL1 = files.Snapshot(ns.halolabel1, files.HaloLabelFile)


    SNAP2, LABEL2 = None, None
    SNAP2 = files.Snapshot(ns.snapfilename2, files.TPMSnapshotFile)
    LABEL2 = files.Snapshot(ns.halolabel2, files.HaloLabelFile)

    N1 = SNAP1.npart
    N2 = SNAP2.npart


    data1 = numpy.empty(len(SNAP1.read("ID",0,N1)), dtype=[
                ('ID', ('i4')), 
                ('Label', ('i4')), 
                ])
    data2 = numpy.empty(len(SNAP2.read("ID",0,N2)), dtype=[
                ('ID', ('i4')), 
                ('Label', ('i4')), 
                ])

    data1['ID'] = SNAP1.read("ID",0,N1)
    data2['ID'] = SNAP2.read("ID",0,N2)
    data1['Label'] = LABEL1.read("Label",0,N1)
    data2['Label'] = LABEL2.read("Label",0,N2)

    sdata1 = numpy.sort(data1, axis=0)
    sdata2 = numpy.sort(data2, axis=0)  

    i = 0

    trim_array = []
    halo_res_array = []

    for i in range(0,numpy.max(data1['Label'])): 

        masked2 = sdata1['Label']== i
        cor_label2 = sdata2['Label'][masked2]
        trim, indices = numpy.unique(cor_label2, return_inverse=True)
        halo_res = numpy.bincount(indices)
        trim_array.append(trim)
        halo_res_array.append(halo_res)

    trim_res = numpy.concatenate(trim_array)
    halo_res = numpy.concatenate(halo_res_array)

    if ns.output != '-':
        ff = open(ns.output, 'w')
    with ff:
        numpy.savetxt(ff, zip(trim_res,halo_res), fmt='%d')
#        numpy.savetxt(ff, zip(sdata1['Label'], sdata1['ID'], sdata2['ID'],  sdata2['Label']), fmt='%d')
main()
