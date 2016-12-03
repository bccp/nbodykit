from nbodykit.core import Algorithm, DataSource, GridSource, Painter
from pmesh.pm import ParticleMesh, RealField, ComplexField

import os
import numpy

class PaintGridAlgorithm(Algorithm):
    """
    Algorithm to paint a data source to a 3D configuration space grid.

    Notes
    -----
    The algorithm saves the grid to a bigfile File.

    """
    plugin_name = "PaintGrid"
    
    def __init__(self, Nmesh, DataSource, Painter=None, paintbrush='cic', paintNmesh=None, 
                    dataset='PaintGrid', Nfile=0, writeFourier=False):
                    
        # set the  input
        self.Nmesh        = Nmesh
        self.datasource   = DataSource
        self.dataset      = dataset
        self.writeFourier = writeFourier 
        
        # set the painter
        if Painter is None:
            Painter = Painter.create("DefaultPainter", paintbrush=paintbrush)
        self.painter = Painter
        self.painter.paintbrush = paintbrush
        
        # Nmesh for the painter
        if paintNmesh is None:
            paintNmesh = self.Nmesh
        self.paintNmesh = paintNmesh

        self.Nfile = Nfile
        if self.Nfile == 0:
            chunksize = 1024 * 1024 * 512
            self.Nfile = (self.Nmesh * self.Nmesh * self.Nmesh + chunksize - 1)// chunksize

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "periodic power spectrum calculator via FFT"

        # required
        s.add_argument("Nmesh", type=int, 
            help='the number of cells in the gridded mesh')

        # the first field
        s.add_argument('DataSource', type=(DataSource.from_config, GridSource.from_config),
            required=True, help="DataSource)")
        s.add_argument('Painter', type=Painter.from_config, required=False, 
            help='the Painter; run `nbkit.py --list-painters` for all options')

        s.add_argument('dataset', help="name of dataset to write to")
        s.add_argument('Nfile', required=False, help="number of files")
        s.add_argument('writeFourier', type=bool, required=False, help="Write complex Fourier modes instead?")
        s.add_argument('paintNmesh', type=int, required=False,
                    help="The painting Nmesh. The grid will be Fourier resampled to Nmesh before output. A value larger than Nmesh can reduce grid artifacts.")
        s.add_argument('paintbrush', type=lambda x: x.lower(), choices=['cic', 'tsc'],
            help='the density assignment kernel to use when painting; '
                 'CIC (2nd order) or TSC (3rd order)')

    def run(self):
        """
        Run the algorithm, which computes and returns the grid in C_CONTIGUOUS order partitioned by ranks.
        """
        from nbodykit import measurestats

        if self.comm.rank == 0:
            self.logger.info('importing done')
            self.logger.info('Resolution Nmesh : %d' % self.paintNmesh)
            self.logger.info('paintbrush : %s' % self.painter.paintbrush)

        # setup the particle mesh object, taking BoxSize from the painters
        pmpaint = ParticleMesh(BoxSize=self.datasource.BoxSize, Nmesh=[self.paintNmesh] * 3, dtype='f4', comm=self.comm)
        pm = ParticleMesh(BoxSize=self.datasource.BoxSize, Nmesh=[self.Nmesh] * 3, dtype='f4', comm=self.comm)

        real, stats = self.painter.paint(pmpaint, self.datasource)

        if self.writeFourier:
            result = ComplexField(pm)
        else:
            result = RealField(pm)
        real.resample(result)

        # reuses the memory
        result.sort(out=result)
        result = result.value.ravel()

        # return all the necessary results
        return result, stats

    def save(self, output, result):
        """
        Save the power spectrum results to the specified output file
        
        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the tuple returned by `run()` -- first argument specifies the bin
            edges and the second is a dictionary holding the data results
        """
        import bigfile
        import numpy

        result, stats = result

        if self.comm.rank == 0:
            self.logger.info('Output Nmesh : %d' % self.Nmesh)
            self.logger.info("writing to %s/%s in %d parts" % (output, self.dataset, self.Nfile))

        f = bigfile.BigFileMPI(self.comm, output, create=True)

        b = f.create_from_array(self.dataset, result, Nfile=self.Nfile)

        b.attrs['ndarray.shape'] = numpy.array([self.Nmesh, self.Nmesh, self.Nmesh], dtype='i8')
        b.attrs['BoxSize'] = numpy.array(self.datasource.BoxSize, dtype='f8')
        b.attrs['Nmesh'] = self.Nmesh
        b.attrs['paintNmesh'] = self.paintNmesh
        b.attrs['paintbrush'] = self.painter.paintbrush
        b.attrs['Ntot'] = stats['Ntot']

