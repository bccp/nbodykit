import logging
from nbodykit import files 
from mpi4py import MPI

class TPMSnapshotPainter(object):
    
    @classmethod
    def register(kls, inputdesc):
        h = inputdesc.add_parser("TPMSnapshot", 
            usage="TPMSnapshot:path[:&rsd=[x|y|z]]")
        h.add_argument("path", help="path to file")
        h.add_argument("&rsd", 
            choices="xyz", help="direction to do redshift distortion")
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