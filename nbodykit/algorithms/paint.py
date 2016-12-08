import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.base.algorithm import Algorithm

# touch the file
class Paint(Algorithm):
    logger = logging.getLogger('Paint')
    
    @CurrentMPIComm.enable
    def __init__(self, field, Nmesh, comm=None):
        from pmesh.pm import ParticleMesh

        self.comm = comm 

        # save meta-data
        self.attrs['Nmesh'] = Nmesh

        self.field = field

        self.pm = ParticleMesh(BoxSize=self.field.BoxSize, Nmesh=[self.attrs['Nmesh']]*3,
                                dtype='f4', comm=self.comm)

        Algorithm.__init__(self, comm)

    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """

        self.results.real = self.field.paint(self.pm).value

