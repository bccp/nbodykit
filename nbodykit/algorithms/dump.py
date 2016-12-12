import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.base.algorithm import Algorithm, Result

class DumpFieldResult(Result):

    @property
    def state(self):
        if hasattr(self, "real"):
            data = numpy.empty(shape=self.real.size, dtype=self.real.dtype)
            self.real.sort(out=data)
            return dict(real=data, Nmesh=self.real.Nmesh, BoxSize=self.real.BoxSize)
        else:
            data = numpy.empty(shape=self.complex.size, dtype=self.complex.dtype)
            self.complex.sort(out=data)
            return dict(complex=data, Nmesh=self.complex.Nmesh, BoxSize=self.complex.BoxSize)

    def save(self, out, dataset="Field"):
        import bigfile
        state = self.state
        with bigfile.BigFileMPI(self.comm, out, create=True) as ff:
            if hasattr(self, "real"):
                with ff.create_from_array(dataset, state['real']) as bb:
                    bb.attrs['ndarray.shape'] = state['Nmesh']
                    bb.attrs['BoxSize'] = state['BoxSize']
                    bb.attrs['Nmesh'] = state['Nmesh']
            else:
                with ff.create_from_array(dataset, state['complex']) as bb:
                    bb.attrs['ndarray.shape'] = state['Nmesh'], state['Nmesh'], state['Nmesh'] // 2 + 1
                    bb.attrs['BoxSize'] = state['BoxSize']
                    bb.attrs['Nmesh'] = state['Nmesh']

class DumpField(Algorithm):
    logger = logging.getLogger('DumpField')
    
    @CurrentMPIComm.enable
    def __init__(self, source, Nmesh, kind="real", comm=None):
        """
        Parameters
        ----------
        kind : string
            "real" or "complex"

        """

        from pmesh.pm import ParticleMesh

        self.comm = comm

        # save meta-data
        self.attrs['Nmesh'] = Nmesh

        self.source = source

        assert kind in ["real", "complex"]

        self.kind = kind

        self.pm = ParticleMesh(BoxSize=self.source.BoxSize, Nmesh=[self.attrs['Nmesh']]*3,
                                dtype='f4', comm=self.comm)

        Algorithm.__init__(self, comm, DumpFieldResult)

    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """

        real = self.source.paint(self.pm)
        if self.kind == 'real':
            self.result.real = real
        elif self.kind == 'complex':
            self.result.complex = real.r2c()

