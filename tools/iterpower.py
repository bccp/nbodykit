
def read_selections(filename):
    """
    Read a parameter file that specifies the `select`
    keyword for `nbodykit::power.py` for a set of samples
    
    Notes
    -----
    *   File format should be :code: key = "condition" for an
        auto-spectrum or :code: key = ["condition1", "condition2"]
        for a cross-spectrum. 
    *   If no selection is needed, use :code: key = None
    *   Keys are treated as the names of the samples
    
    Parameters
    ----------
    filename    : str
        the name of the file holding the parameters
        
    Returns
    -------
    d   : dict
        the dictionary holding the selection condition for 
        each sample key
    """
    toret = {}
    cnt = 0
    for line in open(filename, 'r'):
        if not line.strip() or line.lstrip()[0] == '#': continue
        fields = line.split('=', 1)
        toret[fields[0].strip()] = eval(fields[-1])
        cnt += 1
    return toret
    
def read_power_params(filename):
    """
    Read a parameter file that specifies the parameters
    for `nbodykit::power.py`
    
    Notes
    -----
    *   File format should be :code: key = value
    *   The keys should be:
            mode :
                the mode of `nbodykit::power.py`, one of `1d`,`2d`
            box_size :
                the size of the simulation box
            Ncells : 
                the number of cells on the density grid to use
            output : 
                string specifying the output file. Must have
                `{tag}` in the string, which is replaced by 
                the name of the selected sample
            file1, file2 : 
                the strings specifying the input file, which 
                will have any selection appended to it
    *   Any other parameters in the dictionary are treated
        as options to `nbodykit::power.py`
    
    Parameters
    ----------
    filename    : str
        the name of the file holding the parameters
        
    Returns
    -------
    d   : dict
        the dictionary holding the parameters
    """
    toret = {}
    for line in open(filename, 'r'):
        if not line.strip() or line.lstrip()[0] == '#': continue
        fields = line.split('=', 1)
        toret[fields[0].strip()] = fields[-1].strip()
    
    # check the keys
    necessary = ['mode', 'box_size', 'Ncells', 'output', 'file1']
    if not all(key in toret for key in necessary):
        raise RuntimeError("missing keys in `read_power_params`: need %s" %necessary)
    if 'file2' not in toret:
        toret['file2'] = toret['file1']
    return toret

def write_power_params(tag, power_dict, select_params):
    """
    Write out a parameter file that can be passed to `nbodykit::power.py`
    using the `@filename` syntax
    
    Parameters
    ----------
    tag : str
        the name of sample to select
    power_dict : dict
        dictionary specifying the main power.py parameters
    select_params : dict
        dictionary specifying the `select` option 
        
    Returns
    -------
    fname   : str
        the name of the output file
    """
    import tempfile
    
    params = power_dict.copy()
    if tag not in select_params.keys():
        raise ValueError("specified tag must be one of [%s]" %", ".join(select_params.keys()))
    
    # make the selection param a list
    select = select_params[tag]
    if not isinstance(select, list):
        select = [select]
        
    # write out to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as ff:
        filename = ff.name
        
        # now write out
        ff.write(params.pop('mode')+'\n')
        ff.write(params.pop('box_size')+'\n')
        ff.write(params.pop('Ncells')+'\n')
        if "{tag}" not in params['output']:
            raise RuntimeError("sample tag should be specified in `output` as {tag}")
        ff.write(params.pop('output').replace("{tag}", tag)+'\n')
        
        # now the input files
        for i in range(len(select)):
            file_fmt = params.pop('file%d' %(i+1))
            if select[i] is not None:
                file_fmt += ":-select= %s\n" %(select[i])
            ff.write(file_fmt)
        
        # anything else is an option
        for d, k in params.iteritems():
            ff.write("--%s=%s\n" %(d,k))
        
    return filename
            
    
def parse_args(desc, samples):
    """
    Parse the command line arguments and return the namespace
    
    Parameters
    ----------
    desc : str
        the description for the argument parser
    samples : list of str
        a list of the samples names. these provide the choices
        for the `sample` command line argument, which selects
        which sample we are computing results for
    
    Returns
    -------
    args : argparse.Namespace
        namespace holding the commandline arguments
    """
    import argparse as ap
    
    parser = ap.ArgumentParser(description=desc, 
                formatter_class=ap.ArgumentDefaultsHelpFormatter)
                            
    h = 'the name of the PBS job file to run. This file should take one' + \
        'command line argument specfying the input `power.py` parameter file'
    parser.add_argument('job_file', type=str, help=h)
    h = 'the sample name'
    parser.add_argument('samples', choices=samples+['all'], nargs='+', help=h)
    h = 'the name of the file specifying the main power.py parameters'
    parser.add_argument('-p', '--power', required=True, type=str, help=h)
    h = 'the name of the file specifying the selection parameters'
    parser.add_argument('-s', '--select', required=True, type=str, help=h)
    
    return parser.parse_args()

def qsub_samples(args, samples):
    """
    Submit the job script specified on the command line for the desired 
    sample(s). This could submit several job at once, but will 
    wait 1 second in between doing so.  
    """
    import subprocess
    import time
    
    # determine the sample we are running
    if len(args.samples) == 1 and args.samples[0] == 'all':
        args.samples = samples
        
    # read the params and write out
    s = read_selections(args.select)
    p = read_power_params(args.power)
    
    # submit the jobs
    for sample in args.samples:
        fname = write_power_params(sample, p, s)
        ret = subprocess.call(['qsub', '-v', 'param_file=%s' %fname, args.job_file])
        time.sleep(1)

    
        
    
    
    
    