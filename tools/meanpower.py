"""
    meanpower.py
    designed to read in a set of 1D/2D power plain text
    files and output the mean power to a plain text file
"""
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse as ap
from glob import glob
from nbodykit import files, pkmuresult, plugins
import numpy

desc = "designed to read in a set of 1D or 2D power plain text " + \
       "files and output the mean power to a plain text file"
parser = ap.ArgumentParser(description=desc, 
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
                            
h = 'the mode, either 1D or 2D'
parser.add_argument('mode', choices=['1d', '2d'], help=h)
h = 'the pattern to match files power files on'
parser.add_argument('pattern', type=str, help=h)
h = 'the name of the output file'
parser.add_argument('output', default=None, type=str, help=h)
h = 'loop over these arguments, doing a string replace on the ' + \
    'input/output files, using %s format'
parser.add_argument('--batch', nargs='+', type=str, help=h)

args = parser.parse_args()

def main():

    nbatch = 1
    if args.batch is not None:
        nbatch = len(args.batch)
    
    # do each pattern provided
    for n in range(nbatch):
        
        # try to replace with the batch string
        pattern = args.pattern
        output_file = args.output
        if args.batch is not None:
            print "processing batch string %s..." %args.batch[n]
            
            # do the pattern
            fmt_count = pattern.count('%s')
            if fmt_count > 0:
                fmt_strs = (args.batch[n])*fmt_count
                pattern = pattern %fmt_strs
                
            # and the output file
            fmt_count = output_file.count('%s')
            if fmt_count > 0:
                fmt_strs = (args.batch[n])*fmt_count
                output_file = output_file %fmt_strs
        
        # read the files
        results = glob(pattern)
        print pattern
        if not len(results):
            raise RuntimeError("whoops, no files match input pattern `%s`" %pattern)
    
        # loop over each file
        data = []
        for f in results:
            print f
            print "averaging %d files..." %len(f)
            if args.mode == '2d':
                try:
                    d, meta = files.ReadPower2DPlainText(f)
                except Exception as e:
                    raise RuntimeError("error reading `%s` as a power 2D plain text file: %s" %(f, str(e)))
                data.append(pkmuresult.PkmuResult.from_dict(d, **meta))
            else:
                data.append(numpy.loadtxt(f))
        
        # average and output
        if args.mode == '2d':
            avg = pkmuresult.PkmuResult.from_list(data, sum_only=['modes'], weights='modes')
            output = plugins.Power2DStorage(output_file)
    
            data = {k:avg[k].data for k in avg.columns}
            data['edges'] = [avg.kedges, avg.muedges]
            meta = {k:getattr(avg,k) for k in avg._metadata}
            output.write(data, **meta)
        else:        
            avg = []
            for i, name in zip(range(3), ['k', 'power', 'modes']):
            
                if name == 'modes':
                    avg.append(numpy.sum([d[:,i] for d in data], axis=0))
                else:
                    # mask any element that is NaN for all spectra we are avging
                    mask = numpy.ones(data[0].shape[0], dtype=bool)
                    for d in data:
                        mask &= numpy.isnan(d[:,i])
                
                    colavg = numpy.empty(mask.shape)
                    weights = [d[:,2][~mask] for d in data] # weight by modes
                    colavg[~mask] = numpy.average([d[:,i][~mask] for d in data], axis=0, weights=weights)
                    colavg[mask] = numpy.nan
                    avg.append(colavg)

            output = plugins.Power1DStorage(output_file) 
            output.write(avg)
    
if __name__ == '__main__':
    main()