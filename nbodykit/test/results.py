from . import os, cache_dir
from .. import pkg_dir

def get_result_paths(output_file):
    """
    Return the full reference paths for a given output file name
    """
    this = os.path.join(pkg_dir, 'examples', 'output', output_file)
    ref = os.path.join(os.path.expanduser(cache_dir), 'results', output_file)
    return this, ref
        
def compute_md5sum(path):
    """
    Return the md5 hash for the input path. If the input path is
    a directory, return the concatenatation of all files in the directory
    
    Parameters
    ----------
    path : str
        the name of a file or directory path
    
    Returns
    -------
    str : 
        the md5 sum hash string
    """    
    from pytest_pipeline import utils
    
    if not os.path.exists(path):
        raise ValueError("no result file located at '%s'" %path)
        
    if os.path.isdir(path):
        paths_to_hash = []
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                paths_to_hash.append(os.path.join(root,filename))
        
        return "".join(utils.file_md5sum(f) for f in sorted(paths_to_hash))
        
    else:
        return utils.file_md5sum(path) 
        
def load_pandas_hdf(filename, dataset):
    """
    Load a pandas HDF file, returning the ``DataFrame`` object
    """
    from pandas import read_hdf
    return read_hdf(filename, dataset)
    
def load_hdf(filename, dataset):
    """
    Load a HDF5 file using ``h5py``, returning a ``numpy`` array
    """
    import h5py
    return h5py.File(filename, mode='r')[dataset][...]
    
def get_dataset_loader(dim, stat):
    """
    Load a ``Dataset`` object from file
    """
    from nbodykit import files, dataset
    
    if stat not in ['power', 'corr']:
        raise ValueError("``stat`` in `load_dataset` should be one of ['power', 'corr']")
    if dim not in ['1d', '2d']:
        raise ValueError("``dim`` in `load_dataset` should be one of ['1d', '2d']")
       
    if dim == '2d':
        reader = files.Read2DPlainText
        cls = dataset.Power2dDataSet if stat == 'power' else dataset.Corr2dDataSet
    else:
        reader = files.Read1DPlainText
        cls = dataset.Power1dDataSet if stat == 'power' else dataset.Corr1dDataSet
    
    return lambda f: cls.from_nbkit(*reader(f))

    