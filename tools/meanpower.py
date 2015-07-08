"""
    meanpower.py
    designed to read in a set of 2D power plain text
    files and output the mean power to a plain text file
"""
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse as ap
from glob import glob
from nbodykit import files, pkmuresult, plugins

desc = "designed to read in a set of 2D power plain text " + \
       "files and output the mean power to a plain text file"
parser = ap.ArgumentParser(description=desc, 
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)
                            
h = 'the pattern to match files 2D power files on'
parser.add_argument('pattern', type=str, help=h)
h = 'the name of the output file'
parser.add_argument('output', default=None, type=str, help=h)

args = parser.parse_args()

def main():

    # read the files
    results = glob(args.pattern)
    if not len(results):
        raise RuntimeError("whoops, no files match input pattern")
    
    # loop over each file
    pkmus = []
    for f in results:
        #try:
        d, meta = files.ReadPower2DPlainText(f)
        #except:
        #raise RuntimeError("error reading `%s` as a power 2D plain text file" %f)
        pkmus.append(pkmuresult.PkmuResult.from_dict(d, **meta))
        
    # get the average
    avg = pkmuresult.PkmuResult.from_list(pkmus, sum_only=['modes'], weights='modes')
    
    # output
    output = plugins.Power2DStorage(args.output)
    
    data = {k:avg[k].data for k in avg.columns}
    data['edges'] = [avg.kedges, avg.muedges]
    meta = {k:getattr(avg,k) for k in avg._metadata}
    output.write(data, **meta)
    
if __name__ == '__main__':
    main()