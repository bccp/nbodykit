from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser("Parallel Power Spectrum Calculator",
        formatter_class=RawTextHelpFormatter,
        description=
     """Calculating matter power spectrum from RunPB input files. 
        Output is written to stdout, in Mpc/h units. 
        PowerSpectrum is the true one, without (2 pi) ** 3 factor. (differ from Gadget/NGenIC internal)
        This script moves all particles to the halo center.
     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
        The author would like thank Marcel Schmittfull for the explanation on cic, shotnoise, and k==0 plane errors.
     """
        )

parser.add_argument("mode", choices=["2d", "1d"]) 

parser.add_argument("--binshift", type=float, default=0.0,
        help='Shift the bin center by this fraction of the bin width. Default is 0.0. Marcel uses 0.5. this shall rarely be changed.' )
parser.add_argument("--bunchsize", type=int, default=1024*1024*4,
        help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh')
parser.add_argument("--remove-cic", default='anisotropic', choices=["anisotropic","isotropic", "none"],
        help='deconvolve cic, anisotropic is the proper way, see http://www.personal.psu.edu/duj13/dissertation/djeong_diss.pdf')
parser.add_argument("--Nmu", type=int, default=5,
        help='the number of mu bins to use' )


class InputDesc(object):
    """ describing an input field. 
        Format is 

        1)  HaloFile:path:min:max:m0

        where mmin and mmax are the min/max mass in log10
        m0 is the mass of a particle

        2)  TPMSnapshot:path
    """
    parser = ArgumentParser("")
    subparsers = parser.add_subparsers()
    h = subparsers.add_parser("self")
    h.set_defaults(painter=None)

    def __init__(self, string):
        self.string = string
        words = string.split(':')
        ns = self.parser.parse_args(words)
        self.__dict__.update(ns.__dict__)

    @classmethod
    def add_parser(kls, name, usage):
        return kls.subparsers.add_parser(name, 
                usage=usage, add_help=False)
    @classmethod
    def format_help(kls):
        rt = []
        for k in kls.subparsers.choices:
            rt.append(kls.subparsers.choices[k].format_help())

        return '\n'.join(rt)

class HaloFilePainter(object):
    @classmethod
    def register(kls, inputdesc):
        h = inputdesc.add_parser("HaloFile", 
            usage="HaloFile:path:min:max:m0[:--rsd=[x|y|z]]",
            )
        h.add_argument("path", help="path to file")
        h.add_argument("logMmin", type=float, help="log10 min mass")
        h.add_argument("logMmax", type=float, help="log10 max mass")
        h.add_argument("m0", type=float, help="mass mass of a particle")
        h.add_argument("--rsd", 
            choices="xyz", help="direction to do Redshift distortion")

        h.set_defaults(painter=kls.paint)
    @classmethod
    def paint(kls, ns, desc, pm):
        if pm.comm.rank == 0:
            hf = files.HaloFile(desc.path)
            nhalo = hf.nhalo
            halopos = numpy.float32(hf.read_pos())
            halomass = numpy.float32(hf.read_mass() * desc.m0)
            logmass = numpy.log10(halomass)
            mask = logmass > desc.logMmin
            mask &= logmass < desc.logMmax
            halopos = halopos[mask]
            logging.info("total number of halos in mass range is %d" % mask.sum())
        else:
            halopos = numpy.empty((0, 3), dtype='f4')
            halomass = numpy.empty(0, dtype='f4')

        Ntot = len(halopos)
        Ntot = pm.comm.bcast(Ntot)

        halopos *= ns.BoxSize

        layout = pm.decompose(halopos)
        tpos = layout.exchange(halopos)
        pm.paint(tpos)

        npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
        return Ntot

class TPMSnapshotPainter(object):
    @classmethod
    def register(kls, inputdesc):
        h = inputdesc.add_parser("TPMSnapshot", 
            usage="TPMSnapshot:path[:--rsd=[x|y|z]]")
        h.add_argument("path", help="path to file")
        h.add_argument("--rsd", 
            choices="xyz", help="direction to do Redshift distortion")
        h.set_defaults(painter=kls.paint)
    @classmethod
    def paint(kls, ns, desc, pm):
        pm.real[:] = 0
        Ntot = 0
        for round, P in enumerate(
                files.read(pm.comm, 
                    desc.path, 
                    files.TPMSnapshotFile, 
                    columns=['Position'], 
                    bunchsize=ns.bunchsize)):

            nread = pm.comm.allreduce(len(P['Position']), op=MPI.SUM) 

            P['Position'] *= ns.BoxSize
            layout = pm.decompose(P['Position'])
            tpos = layout.exchange(P['Position'])
            #print tpos.shape
            pm.paint(tpos)
            npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
            if pm.comm.rank == 0:
                logging.info('round %d, npaint %d, nread %d' % (round, npaint, nread))
            Ntot = Ntot + nread
        return Ntot

TPMSnapshotPainter.register(InputDesc) 
HaloFilePainter.register(InputDesc) 

parser.add_argument("BoxSize", type=float, 
        help='BoxSize in Mpc/h')
parser.add_argument("Nmesh", type=int, 
        help='size of calculation mesh, recommend 2 * Ngrid')
parser.add_argument("output", help='write power to this file') 

parser.add_argument("input1", type=InputDesc, help=InputDesc.format_help())
parser.add_argument("input2", type=InputDesc, help="see input1", default=None)

ns = parser.parse_args()

# done with the parser. now do the real calculation

logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit import files 
from nbodykit.measurepower import measure2Dpower, measurepower

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction


from mpi4py import MPI

def main():

    if MPI.COMM_WORLD.rank == 0:
        print 'importing done'

    pm = ParticleMesh(ns.BoxSize, ns.Nmesh, dtype='f4')

    Ntot1 = ns.input1.painter(ns, ns.input1, pm)

    if MPI.COMM_WORLD.rank == 0:
        print 'painting done'
    pm.r2c()
    if MPI.COMM_WORLD.rank == 0:
        print 'r2c done'

    if ns.input2.painter is not None:
        # cross power 
        complex = pm.complex.copy()
        numpy.conjugate(complex, out=complex)

        Ntot2 = ns.input2.painter(ns, ns.input2, pm)
        if MPI.COMM_WORLD.rank == 0:
            print 'painting 2 done'
        pm.r2c()
        if MPI.COMM_WORLD.rank == 0:
            print 'r2c 2 done'
        complex *= pm.complex
        complex **= 0.5

        if MPI.COMM_WORLD.rank == 0:
            print 'cross done'
    else:
        # auto power 
        complex = pm.complex
    
    if ns.mode == "1d":
        do1d(pm, complex, ns)

    if ns.mode == "2d":
        do2d(pm, complex, ns)
    
def do2d(pm, complex, ns):
    k, mu, p, N, edges = measure2Dpower(pm, complex, ns.binshift, ns.remove_cic, 0, ns.Nmu)
  
    if MPI.COMM_WORLD.rank == 0:
        print 'measure'

    if pm.comm.rank == 0:
        if ns.output != '-':
            myout = open(ns.output, 'w')
        else:
            myout = stdout
        numpy.savetxt(myout, zip(k.flat, mu.flat, p.flat, N.flat), '%0.7g')
        myout.flush()

def do1d(pm, complex, ns):
    k, p = measurepower(pm, complex, ns.binshift, ns.remove_cic, 0)

    if MPI.COMM_WORLD.rank == 0:
        print 'measure'

    if pm.comm.rank == 0:
        if ns.output != '-':
            myout = open(ns.output, 'w')
        else:
            myout = stdout
        numpy.savetxt(myout, zip(k, p), '%0.7g')
        myout.flush()

main()
