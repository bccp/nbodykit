import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.base.algorithm import Algorithm, Result

class PaintResult(Result):

    @property
    def state(self):
        data = numpy.empty(shape=self.real.size, dtype=self.real.dtype)
        self.real.sort(out=data)
        return dict(real=data, Nmesh=self.real.Nmesh, BoxSize=self.real.BoxSize)

    def save(self, out, dataset="Field"):
        import bigfile
        state = self.state
        with bigfile.BigFileMPI(self.comm, out, create=True) as ff:
            with ff.create_from_array(dataset, state['real']) as bb:
                bb.attrs['ndarray.shape'] = state['Nmesh']
                bb.attrs['BoxSize'] = state['BoxSize']

class Paint(Algorithm):
    logger = logging.getLogger('Paint')
    
    @CurrentMPIComm.enable
    def __init__(self, source, Nmesh, comm=None):
        from pmesh.pm import ParticleMesh

        self.comm = comm

        # save meta-data
        self.attrs['Nmesh'] = Nmesh

        self.source = source

        self.pm = ParticleMesh(BoxSize=self.source.BoxSize, Nmesh=[self.attrs['Nmesh']]*3,
                                dtype='f4', comm=self.comm)

        Algorithm.__init__(self, comm, PaintResult)

    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """

        real = self.source.paint(self.pm)

        self.result.real = real

