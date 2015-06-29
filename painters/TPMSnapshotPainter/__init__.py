import logging
from nbodykit import files 
from mpi4py import MPI

class TPMSnapshotPainter(object):
    
    def __init__(self, data):
        self.__dict__.update(data.__dict__)
    
    @classmethod
    def register(kls, inputdesc):
        h = inputdesc.add_parser("TPMSnapshot", 
            usage="TPMSnapshot:path[:&rsd=[x|y|z]]")
        h.add_argument("path", help="path to file")
        h.add_argument("&rsd", 
            choices="xyz", default=None, help="direction to do redshift distortion")
        h.set_defaults(klass=kls)

    def paint(self, ns, pm):
        pm.real[:] = 0
        Ntot = 0
        for round, P in enumerate(
                files.read(pm.comm, 
                    self.path, 
                    files.TPMSnapshotFile, 
                    columns=['Position', 'Velocity'], 
                    bunchsize=ns.bunchsize)):

            nread = pm.comm.allreduce(len(P['Position']), op=MPI.SUM) 

            if self.rsd is not None:
                dir = "xyz".index(self.rsd)
                P['Position'][:, dir] += P['Velocity'][:, dir]

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