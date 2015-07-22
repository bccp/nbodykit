from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd


desc = "compute galaxy subtypes (A/B) for MBII central and satellite populations"
parser = ArgumentParser(desc)

# add the positional arguments
parser.add_argument("path", help="path to MBII simulation files")
parser.add_argument("simulation", help="name of simulation", choices=["dmo", "mb2"])
args = parser.parse_args()


def add_halo_sizes(df):
    """
    Compute number of centrals/satellites per halo and add to the 
    input DataFrame as a column
    """
    # delete N_sat, N_cen columns if they exist
    if 'N_sat' in df.columns: del df['N_sat']
    if 'N_cen' in df.columns: del df['N_cen']
    
    # these are grouped by type and then by halo id
    halos = df.groupby(['type', 'haloid'])
    
    # the sizes
    sizes = halos.size()
    
    # add N_cen if there are any centrals in this sample
    if 'central' in sizes.index.levels[0]:
        N_cen = pd.DataFrame(sizes['central'], columns=['N_cen'])
        df = df.join(N_cen, on='haloid', how='left')
        
        # now fill missing values with zeros
        df.N_cen.fillna(value=0., inplace=True)
    
    # add the satellites
    if 'satellite' in sizes.index.levels[0]:
        N_sat = pd.DataFrame(sizes['satellite'], columns=['N_sat'])
        df = df.join(N_sat, on='haloid', how='left')

        # now fill missing values with zeros
        df.N_sat.fillna(value=0., inplace=True)
    
    return df
        
        
def main():
    
    # load sat and cen halos
    path_args = (args.path, args.simulation)
    haloid_cen = np.fromfile("{}/Centrals/{}_cenHaloID".format(*path_args), dtype=('i8'))
    haloid_sat = np.fromfile("{}/Satellites/{}_satHaloID".format(*path_args), dtype=('i8'))
    haloids = np.concatenate([haloid_cen, haloid_sat])
    
    # make the dataframe
    types = np.concatenate([np.repeat(['central'], len(haloid_cen)), np.repeat(['satellite'], len(haloid_sat))])
    data = {'type' : types, 'haloid' : haloids}
    df = pd.DataFrame(data=data)
    
    # add halo sizes
    df = add_halo_sizes(df)
    
    # central subtypes
    df.loc[(df.type == 'central')&(df.N_sat == 0), 'subtype'] = 'A'
    df.loc[(df.type == 'central')&(df.N_sat > 0), 'subtype'] = 'B'
    
    # satellite subtypes
    df.loc[(df.type == 'satellite')&(df.N_sat == 1), 'subtype'] = 'A'
    df.loc[(df.type == 'satellite')&(df.N_sat > 1), 'subtype'] = 'B'
    
    # split back into cens and sats
    df_cen = df.loc[df.type == 'central']
    df_sat = df.loc[df.type == 'satellite']
    
    # write out subtypes
    df_cen.subtype.values.astype('S1').tofile('{}/Centrals/{}_subtype'.format(*path_args))
    df_sat.subtype.values.astype('S1').tofile('{}/Satellites/{}_subtype'.format(*path_args))

if __name__ == '__main__':
    main()