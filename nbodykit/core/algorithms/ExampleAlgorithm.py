from nbodykit.core import Algorithm, Source, DataSource
import numpy
from pmesh.pm import ParticleMesh

class Play(Algorithm):
    """
    A simple example Algorithm that plays the 'read' interface
    of a Source and does min/max on selected columns
    """
    plugin_name = "Play"

    def __init__(self, source, columns=None):

        self.source = source
        if columns is None:
            columns = self.source.columns
        self.columns = columns

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "describe a specific column of the input DataSource"
        s.add_argument("source", type=Source.from_config, 
            help='the DataSource to describe; run `nbkit.py --list-datasources` for all options')
        s.add_argument("columns", type=str, nargs='+',
            help='the column in the DataSource to describe')
     
    def run(self):
        """
        Run the algorithm, which does nothing
        """
        result = {}

        self.logger.info(self.columns)
        mins = map(lambda x : x.min(axis=0), self.source.read(self.columns))
        maxes = map(lambda x : x.max(axis=0), self.source.read(self.columns))

        x = []
        for min, max in zip(mins, maxes):
            x.append(min)
            x.append(max)

        peaks = Source.compute(*x)

        for column, left, right in zip(self.columns, peaks[::2], peaks[1::2]):

            left = numpy.min(self.comm.allgather(left), axis=0)
            right = numpy.max(self.comm.allgather(right), axis=0)

            result[column] = (left, right)

            if self.comm.rank == 0:
                self.logger.info("Column %s: %s - %s" % (column, left, right))

        pm = ParticleMesh(BoxSize=self.source.BoxSize, Nmesh=[128] * 3, dtype='f4', comm=self.comm)

        real = self.source.paint(pm)

        mean = pm.comm.allreduce(real.sum()) / pm.Nmesh.prod()
        var = pm.comm.allreduce(((real - mean)** 2).sum()) / pm.Nmesh.prod()

        if self.comm.rank == 0:
            self.logger.info("Mean of Real = %g, Variance of real = %g" % (mean, var))
            self.logger.info("shotnoise level = %g" % (real.shotnoise))

    def save(self, output, result):
        pass

class Describe(Algorithm):
    """
    A simple example Algorithm that loads a specific column
    from a DataSource and prints the min/max of the column
    """
    plugin_name = "Describe"

    def __init__(self, datasource, column='Position'):
        
        self.datasource = datasource
        self.column     = column

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "describe a specific column of the input DataSource"
        s.add_argument("datasource", type=DataSource.from_config, 
            help='the DataSource to describe; run `nbkit.py --list-datasources` for all options')
        s.add_argument("column", type=str, 
            help='the column in the DataSource to describe')
     
    def run(self):
        """
        Run the algorithm, which does nothing
        """
        left = []
        right = []
        with self.datasource.open() as stream:
            for [col] in stream.read([self.column]):
                left.append(numpy.min(col, axis=0))
                right.append(numpy.max(col, axis=0))
            left = numpy.min(left, axis=0)
            right = numpy.max(right, axis=0)
            left = numpy.min(self.comm.allgather(left), axis=0)
            right = numpy.max(self.comm.allgather(right), axis=0)
        return left, right

    def save(self, output, data):
        left, right = data
        if self.comm.rank == 0:
            template = "DataSource %s Column %s : min = %s max = %s\n"
            if output == '-' or output is None:
                import sys
                output = sys.stdout
            else:
                output = open(output, 'w')
            args = (self.datasource.plugin_name, self.column, str(left), str(right))
            output.write(template %args)

