from .base import FileType
from . import tools
from ..extern.six import string_types

import h5py
import numpy
import os
from collections import  namedtuple
 
ColumnInfo = namedtuple('ColumnInfo', ['size', 'dtype', 'dset'])

def find_datasets(info, attrs, name, obj):
    """
    Recursively add a ``ColumnInfo`` named tuple to the ``info`` dict
    if ``obj`` is a Dataset
    
    When ``obj`` is a structured array with named fields, a
    ``ColumnInfo`` tuple will be added for each of the named fields
    """          
    # only gather info on dataset    
    if isinstance(obj, h5py.Dataset):
        
        # update meta-data (remember: all strings in h5py stored encoded data)
        attrs[str(name)] = {str(k):obj.attrs[k] for k in obj.attrs}
        
        # structured array
        if obj.dtype.kind == 'V':
            for col in obj.dtype.names:
                size = len(obj)
                dtype = obj.dtype[col]
                key = str(os.path.join(name, col))
                info[key] = ColumnInfo(size=size, dtype=dtype, dset=name)
        # normal array
        else:
            size = obj.shape[0]
            subshape = obj.shape[1:]
            fmt = obj.dtype.type
            if len(subshape): fmt = (fmt,) + subshape
            dtype = numpy.dtype(fmt)
            key = str(name)
            info[key] = ColumnInfo(size=size, dtype=dtype, dset=name)
            
class HDFFile(FileType):
    """
    A file object to handle the reading of columns of data from 
    a :mod:`h5py` HDF5 file.
    """    
    def __init__(self, path, root='/', exclude=[]):
        """
        Parameters
        ----------
        path : str
            the file path to load
        root : str, optional
            the start path in the HDF file, loading all data below this path
        exclude : list of str, optional
            list of path names to exclude; these can be absolute paths, or paths
            relative to ``root``
        """
        self.path = path
        self.root = root
        self.attrs = {}
                
        # gather dtype and size information from file
        info = {}
        with h5py.File(self.path, 'r') as ff:
            
            # make sure root and any excluded paths are valid
            if root not in ff:
                raise ValueError("'%s' is not a valid path in HDF file" %root)
            
            # verify and format the excluded names
            _exclude = [] 
            for excluded in exclude:
                if excluded not in ff:
                    if os.path.join(self.root, excluded) not in ff:
                        raise ValueError("'%s' is not a valid path name; cannot be excluded" %excluded)
                    else:
                        excluded = os.path.join(self.root, excluded)
                _exclude.append(excluded.lstrip('/'))
                                    
            # get the info about possible columns
            sub = ff[root]
            if isinstance(sub, h5py.Dataset):
                find_datasets(info, self.attrs, '', sub)
            else:
                sub.visititems(lambda *args: find_datasets(info, self.attrs, *args))
                        
        # exclude columns
        for col in list(info):
            absname = os.path.join(self.root, col)
            if any(absname.lstrip('/').startswith(ex) for ex in _exclude):
                self.logger.info("ignoring excluded column '%s'" %col)
                info.pop(col)
                
        # verify all the datasets have a single size
        sizes = set([info[col].size for col in info])
        if len(sizes) > 1:
            msg = "size mismatch in datasets of file; please use ``exclude`` to remove datasets of the wrong size\n"
            msg += "\n".join(["size of '%s': %d" %(col, info[col].size) for col in info])
            raise ValueError(msg)
        
        # empty file check
        if not len(sizes):
            raise ValueError("HDF file appears to contain datasets")
        self.size = list(sizes)[0]
    
        # if single Dataset with structured array, allow relative names
        unique_dsets = set([info[col].dset for col in info])
        single_structured_arr = len(unique_dsets) == 1 and len(info) > 1
        
        # construct the data type from "info"
        dtype = []
        for col in info:
            name = col
            if single_structured_arr:
                name = name.rsplit('/', 1)[-1]
            dtype.append((name, info[col].dtype))
        self.dtype = numpy.dtype(dtype)
        
        # set the root properly if columns stored as single structured array
        if single_structured_arr:
            name = list(unique_dsets)[0]
            self.root = os.path.join(self.root, name)
            self.attrs = self.attrs[name]
            self.logger.info("detected single structured array stored as dataset; changing root of HDF file to %s" %self.root)
        
    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range
        
        'start' and 'stop' should be between 0 and :attr:`size`,
        which is the total size of the file
        
        Parameters
        ----------
        columns : str, list of str
            the name of the column(s) to return
        start : int
            the row integer to start reading at
        stop : int
            the row integer to stop reading at
        step : int, optional
            the step size to use when reading; default is 1
        
        Returns
        -------
        numpy.array
            structured array holding the requested columns over
            the specified range of rows
        """ 
        if isinstance(columns, string_types): columns = [columns]
        
        dt = [(col, self.dtype[col]) for col in columns]
        toret = numpy.empty(tools.get_slice_size(start, stop, step), dtype=dt)
          
        with h5py.File(self.path, 'r') as ff:
            # compile a list of datasets
            dsets = {}

            for col in columns:
                
                # absolute name of column (with root path prepended)
                name = os.path.join(self.root, col)
                
                if name in ff:
                    # data from a h5py Dataset directly
                    dsets[name] = [(col, None)]
                    continue
                else:
                    # data from a column in a structured array
                    splitcol = name.rsplit('/', 1)
                    if len(splitcol) != 2:
                        raise ValueError("error trying to access column '%s' in HDF file" %col)

                    name, field = splitcol

                    try:
                        dsets[name].append((col, field))
                    except KeyError:
                        dsets[name] = [(col, field)]
            # then read through the list of datasets,
            # columns in the same dataset is read only once.
            # see, http://docs.h5py.org/en/latest/high/dataset.html#reading-writing-data

            # it is a bit ugly seems to work will see if this fixes the slowness.
            for name, cols in dsets.items():
                dset = ff[name]
                if len(cols) == 1 and cols[0][1] is None:
                    [[col, field]] = cols
                    toret[col][:] = dset[start:stop:step]
                else:
                    fields = [field for col, field in cols]
                    fields.append(slice(start, stop, step))
                    results = dset[tuple(fields)]

                    for col, field in cols:
                        if len(cols) > 1:
                            toret[col][:] = results[field]
                        else:
                            toret[col][:] = results

        return toret