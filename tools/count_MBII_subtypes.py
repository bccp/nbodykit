from argparse import ArgumentParser
import numpy as np
import os
import pandas as pd


desc = "count galaxy subtypes for MBII central and satellite populations"
parser = ArgumentParser(desc)

# add the positional arguments
parser.add_argument("path", help="path to MBII simulation files")
parser.add_argument("simulation", help="name of simulation", choices=["dmo", "mb2"])
parser.add_argument("--min_logmass", type=float, help="the minimum log10 mass to include")
parser.add_argument("--max_logmass", type=float, help="the maximum log10 mass to include")
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

    # read mass
    mass_cen = np.log10(np.fromfile("{}/Centrals/{}_mass".format(*path_args), dtype=('f8')))
    mass_sat = np.log10(np.fromfile("{}/Satellites/{}_mass".format(*path_args), dtype=('f8')))
    logmass = np.concatenate([mass_cen, mass_sat])
    
    # make the dataframe
    types = np.concatenate([np.repeat(['central'], len(haloid_cen)), np.repeat(['satellite'], len(haloid_sat))])
    data = {'type' : types, 'haloid' : haloids, 'logmass' : logmass}
    df = pd.DataFrame(data=data)
    
    # add halo sizes
    df = add_halo_sizes(df)
    
    mask = np.ones(len(df), dtype=bool)
    if args.min_logmass is not None:
        mask &= (df.logmass >= args.min_logmass)
    if args.max_logmass is not None:
        mask &= (df.logmass <= args.max_logmass)
    df = df.loc[mask]
    
    # central subtypes
    df.loc[(df.type == 'central')&(df.N_sat == 0), 'subtype'] = 'A'
    df.loc[(df.type == 'central')&(df.N_sat > 0), 'subtype'] = 'B'
    
    # satellite subtypes
    df.loc[(df.type == 'satellite')&(df.N_sat == 1), 'subtype'] = 'A'
    df.loc[(df.type == 'satellite')&(df.N_sat > 1), 'subtype'] = 'B'
    
    # total number of centrals/satellites
    Ncen = 1.*(df.type == 'central').sum()
    Nsat = 1.*(df.type == 'satellite').sum()
    Ngal = Ncen + Nsat
    
    # satellite fraction
    fsat = Nsat / Ngal

    # cen B fraction
    NcB = 1.*((df.type == 'central')&(df.subtype == 'B')).sum()
    fcB = NcB / Ncen
    
    # sat B fraction
    NsB = 1.*((df.type == 'satellite')&(df.subtype == 'B')).sum()
    fsB = NsB / Nsat
    
    print "satellite fraction: N_sat / N_gal = %.5f" %fsat
    print "central type B fraction: N_cenB / N_cen = %.5f" %fcB
    print "satellite type B fraction: N_satB / N_sat = %.5f" %fsB
    
    
if __name__ == '__main__':
    main()