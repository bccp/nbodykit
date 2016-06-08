from nbodykit.extensionpoints import Algorithm, DataSource

import numpy
import logging

class Describe(Algorithm):
    plugin_name = "Describe"
    logger = logging.getLogger(plugin_name)

    def __init__(self, datasource, column='Position'):
        pass

    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "describe a specific column of the data source"
        s.add_argument("datasource", type=DataSource.from_config, 
            help='the DataSource plugin; run `nbkit.py --list-datasources` for all options')
        s.add_argument("column", type=str, help='the column to describe')
     
    def finalize_attributes(self):
        pass

    def run(self):
        """
        Run the algorithm, which does nothing
        """
        left = []
        right = []
        with self.datasource.open() as stream:
            for [pos] in stream.read([self.column]):
                left.append(numpy.min(pos, axis=0))
                right.append(numpy.max(pos, axis=0))
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
                output = file(output, 'w')
            output.write(template % 
                (self.datasource, self.column, str(left), str(right)))

