from nbodykit.extensionpoints import Algorithm
from nbodykit.extensionpoints import DataSource, GridSource, Painter
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
    
    def __init__(self, Nmesh, DataSource, Painter=None, paintbrush='cic'):
        # combine the two fields
        self.datasource = DataSource
        if Painter is None:
            Painter = Painter.create("DefaultPainter")
        self.painter = Painter

    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "periodic power spectrum calculator via FFT"

        # required
        s.add_argument("Nmesh", type=int, 
            help='the number of cells in the gridded mesh')

        # the first field
        s.add_argument('DataSource', type=DataSource.from_config, required=True,
            help="first data field; a tuple of (DataSource, Painter, Transfer)")
        s.add_argument('Painter', type=Painter.from_config, required=False, 
            help='the Painter; run `nbkit.py --list-painters` for all options')

        s.add_argument('paintbrush', type=lambda x: x.lower(), choices=['cic', 'tsc'],
            help='the density assignment kernel to use when painting; '
                 'CIC (2nd order) or TSC (3rd order)')

    def run(self):
        """
        Run the algorithm, which computes and returns the grid
        """
        from nbodykit import measurestats
        from pmesh.particlemesh import ParticleMesh

        if self.comm.rank == 0:
            self.logger.info('importing done')

        # setup the particle mesh object, taking BoxSize from the painters
        pm = ParticleMesh(self.datasource.BoxSize, self.Nmesh, 
                            paintbrush=self.paintbrush, dtype='f4', comm=self.comm)

        self.painter.paint(pm, self.datasource)
        # return all the necessary results
        return pm

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
        import mpsort
        pm = result
        x3d = pm.real.copy()
        istart = pm.partition.local_i_start
        ind = numpy.zeros(x3d.shape, dtype='i8')
        for d in range(3):
            i = numpy.arange(istart[d], istart[d] + x3d.shape[d])
            i = i.reshape([-1 if dd == d else 1 for dd in range(3)])
            ind[...] *= pm.Nmesh
            ind[...] += i

        x3d = x3d.ravel()
        ind = ind.ravel()
        mpsort.sort(x3d, orderby=ind, comm=self.comm)
        if self.comm.rank == 0:
            self.logger.info("writing to %s" % output)

        f = bigfile.BigFileMPI(self.comm, output, create=True)
        b = f.create_from_array('PaintGrid', x3d, Nfile=1)
        b.attrs['ndarray.shape'] = numpy.array([self.Nmesh, self.Nmesh, self.Nmesh], dtype='i8')
        b.attrs['BoxSize'] = numpy.array(self.datasource.BoxSize, dtype='f8')
        b.attrs['Nmesh'] = self.Nmesh
