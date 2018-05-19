import numpy
import logging
import kdcount
import mpsort
import pandas as pd
from six import string_types
import warnings

from nbodykit import CurrentMPIComm
from nbodykit.source.catalog import ArrayCatalog

class CylindricalGroups(object):
    """
    Compute groups of objects using a cylindrical grouping method. We identify
    all satellites within a given cylindrical volume around a central object.

    Results are computed when the object is inititalized, and the result is
    stored in the :attr:`groups` attribute; see the documenation of
    :func:`~CylindricalGroups.run`.

    Input parameters are stored in the :attr:`attrs` attribute dictionary.

    Parameters
    ----------
    source : subclass of :class:`~nbodykit.base.catalog.CatalogSource`
        the input source of particles providing the 'Position' column; the
        grouping algorithm is run on this catalog
    rperp : float
        the radius of the cylinder in the sky plane (i.e., perpendicular
        to the line-of-sight)
    rpar : float
        the radius along the line-of-sight direction; this is 1/2 the
        height of the cylinder
    rankby : str, list, ``None``
        a single or list of column names to rank order the input source by
        before computing the cylindrical groups, such that objects ranked first
        are marked as CGM centrals; if ``None`` is supplied, no sorting will
        be done
    flat_sky_los : bool, optional
        a unit vector of length 3 providing the line-of-sight direction,
        assuming a fixed line-of-sight across the box, e.g., [0,0,1] to use
        the z-axis. If ``None``, the observer at (0,0,0) is used to compute
        the line-of-sight for each pair
    periodic : bool, optional
        whether to use periodic boundary conditions
    BoxSize : float, 3-vector, optional
        the size of the box of the input data; must be provided as
        a keyword or in ``source.attrs`` if ``periodic=True``

    References
    ----------
    Okumura, Teppei, et al. "Reconstruction of halo power spectrum from
    redshift-space galaxy distribution: cylinder-grouping method and halo
    exclusion effect", arXiv:1611.04165, 2016.
    """
    logger = logging.getLogger('CylindricalGroups')

    def __init__(self, source, rankby, rperp, rpar, flat_sky_los=None,
                    periodic=False, BoxSize=None):

        if 'Position' not in source:
            raise ValueError("the 'Position' column must be defined in the input source")

        if rankby is None:
            rankby = []

        if isinstance(rankby, string_types):
            rankby = [rankby]

        for col in rankby:
            if col not in source:
                raise ValueError("cannot rank by column '%s'; no such column" %col)

        self.source = source
        self.comm = source.comm
        self.attrs = {}

        # need BoxSize
        self.attrs['BoxSize'] = numpy.empty(3)
        BoxSize = source.attrs.get('BoxSize', BoxSize)
        if periodic and BoxSize is None:
            raise ValueError("please specify a BoxSize if using periodic boundary conditions")
        self.attrs['BoxSize'][:] = BoxSize

        # LOS must be unit vector
        if flat_sky_los is not None:
            if numpy.isscalar(flat_sky_los) or len(flat_sky_los) != 3:
                raise ValueError("line-of-sight ``flat_sky_los`` should be vector with length 3")
            if not numpy.allclose(numpy.einsum('i,i', flat_sky_los, flat_sky_los), 1.0, rtol=1e-5):
                raise ValueError("line-of-sight ``flat_sky_los`` must be a unit vector")

        # warn if periodic and LOS is None
        if flat_sky_los is None and periodic:
            warnings.warn(("CylindricalGroups using periodic boundary conditions "
                           "with line-of-sight computed from origin (0,0,0); maybe specify a line-of-sight?"))

        # save meta-data
        self.attrs['rpar'] = rpar
        self.attrs['rperp'] = rperp
        self.attrs['periodic'] = periodic
        self.attrs['rankby'] = rankby
        self.attrs['flat_sky_los'] = flat_sky_los

        # log some info
        if self.comm.rank == 0:
            args = (str(rperp), str(rpar))
            self.logger.info("finding groups with rperp=%s and rpar=%s " %args)
            if flat_sky_los is None:
                self.logger.info("  using line-of-sight computed using observer at origin (0,0,0)")
            else:
                self.logger.info("  using line-of-sight vector %s" %str(flat_sky_los))
            msg = "periodic" if periodic else "non-periodic"
            msg = "  using %s boundary conditions" %msg
            if self.attrs['BoxSize'] is not None:
                msg += " (BoxSize = %s)" %str(self.attrs['BoxSize'])
            self.logger.info(msg)


        self.run()

    def run(self):
        """
        Compute the cylindrical groups, saving the results to the
        :attr:`groups` attribute

        Attributes
        ----------
        groups : :class:`~nbodykit.source.catalog.array.ArrayCatalog`
            a catalog holding the result of the grouping. The length of the
            catalog is equal to the length of the input size, i.e., the length
            is equal to the :attr:`size` attribute. The relevant fields are:

            #. cgm_type :
                a flag specifying the type for each object,
                with 0 specifying CGM central and 1 denoting CGM satellite
            #. cgm_haloid :
                The index of the CGM object this object belongs to; an integer
                between 0 and the total number of CGM halos
            #. num_cgm_sats :
                The number of satellites in the CGM halo
        """
        from pmesh.domain import GridND
        from nbodykit.algorithms.fof import split_size_3d

        comm = self.comm
        rperp, rpar = self.attrs['rperp'], self.attrs['rpar']
        rankby = self.attrs['rankby']

        if self.attrs['periodic']:
            boxsize = self.attrs['BoxSize']
        else:
            boxsize = None

        np = split_size_3d(self.comm.size)
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(np))

        # add a column for original index
        self.source['origind'] = self.source.Index

        # sort the data
        data = self.source.sort(self.attrs['rankby'], usecols=['Position', 'origind'])

        # add a column to track sorted index
        data['sortindex'] = data.Index

        # global min/max across all ranks
        pos = data.compute(data['Position'])
        posmin = numpy.asarray(comm.allgather(pos.min(axis=0))).min(axis=0)
        posmax = numpy.asarray(comm.allgather(pos.max(axis=0))).max(axis=0)

        # domain decomposition
        grid = [
            numpy.linspace(posmin[0], posmax[0], np[0] + 1, endpoint=True),
            numpy.linspace(posmin[0], posmax[1], np[1] + 1, endpoint=True),
            numpy.linspace(posmin[0], posmax[2], np[2] + 1, endpoint=True),
        ]
        domain = GridND(grid, comm=comm)

        # run the CGM algorithm
        groups = cgm(comm, data, domain, rperp, rpar, self.attrs['flat_sky_los'], boxsize)

        # make the final structured array
        self.groups = ArrayCatalog(groups, comm=self.comm, **self.attrs)

        # log some info
        N_cen = (groups['cgm_type']==0).sum()
        isolated_N_cen = ((groups['cgm_type']==0)&(groups['num_cgm_sats']==0)).sum()
        N_cen = self.comm.allreduce(N_cen)
        isolated_N_cen = self.comm.allreduce(isolated_N_cen)
        if self.comm.rank == 0:
            self.logger.info("found %d CGM centrals total" %N_cen)
            self.logger.info("%d/%d are isolated centrals (no satellites)" % (isolated_N_cen,N_cen))

        # delete the column we added to source
        del self.source['origind']

def cgm(comm, data, domain, rperp, rpar, los, boxsize):
    """
    Perform the cylindrical grouping method

    This outputs a structured array with the same length as the input data
    with the following fields for each object in the original data:

    #. cgm_type :
        a flag specifying the type for each object,
        with 0 specifying CGM central and 1 denoting CGM satellite
    #. cgm_haloid :
        The index of the CGM object this object belongs to; an integer
        between 0 and the total number of CGM halos
    #. num_cgm_sats :
        The number of satellites in the CGM halo

    Parameters
    ----------
    comm :
        the MPI communicator
    data : CatalogSource
        catalog with sorted input data, including Position
    domain :
        the domain decomposition
    rperp, rpar : float
        the maximum distances to group objects together in the directions
        perpendicular and parallel to the line-of-sight; the cylinder
        has radius ``rperp`` and height ``2 * rpar``
    los :
        the line-of-sight vector
    boxsize :
        the boxsize, or ``None`` if not using periodic boundary conditions
    """
    # whether we do periodic boundary conditions
    periodic = boxsize is not None
    flat_sky = los is not None

    # the maximum distance still inside the cylinder set by rperp,rpar
    rperp2 = rperp**2; rpar2 = rpar**2
    rmax = (rperp2 + rpar2)**0.5

    pos0, origind0, sortindex0 = data.compute(data['Position'], data['origind'], data['sortindex'])

    layout1    = domain.decompose(pos0, smoothing=0)
    pos1       = layout1.exchange(pos0)
    origind1   = layout1.exchange(origind0)
    sortindex1 = layout1.exchange(sortindex0)

    # exchange particles across ranks, accounting for smoothing radius
    layout2    = domain.decompose(pos1, smoothing=rmax)
    pos2       = layout2.exchange(pos1)
    origind2   = layout2.exchange(origind1)
    sortindex2 = layout2.exchange(sortindex1)
    startrank  = layout2.exchange(numpy.ones(len(pos1), dtype='i4')*comm.rank)

    # make the KD-tree
    tree1 = kdcount.KDTree(pos1, boxsize=boxsize).root
    tree2 = kdcount.KDTree(pos2, boxsize=boxsize).root

    dataframe = []
    j_gt_i = numpy.zeros(len(pos1), dtype='f4')
    wrong_rank = numpy.zeros(len(pos1), dtype='f4')

    def callback(r, i, j):

        r1 = pos1[i]
        r2 = pos2[j]
        dr = r1 - r2

        # enforce periodicity in dpos
        if periodic:
            for axis, col in enumerate(dr.T):
                col[col > boxsize[axis]*0.5] -= boxsize[axis]
                col[col <= -boxsize[axis]*0.5] += boxsize[axis]

        # los distance
        if flat_sky:
            rlos2 =  numpy.einsum("ij,j->i", dr, los)**2
        else:
            center = 0.5 * (r1 + r2)
            dot2 = numpy.einsum('ij, ij->i', dr, center)**2
            center2 = numpy.einsum('ij, ij->i', center, center)
            rlos2 = dot2 / center2

        # sky
        dr2 = numpy.einsum('ij, ij->i', dr, dr)
        rsky2 = numpy.abs(dr2 - rlos2)

        # save the valid pairs
        # To Be Valid: pairs must be within cylinder (compare rperp and rpar)
        valid = (rsky2 <= rperp2)&(rlos2 <= rpar2)
        i = i[valid]; j = j[valid];

        # the correctly sorted indices of particles
        sort_i = sortindex1[i]
        sort_j = sortindex2[j]

        # the rank where the j object lives
        rank_j = startrank[j]

        # track pairs where sorted j > sorted i
        weights = numpy.where(sort_i < sort_j, 1, 0)
        j_gt_i[:] += numpy.bincount(i, weights=weights, minlength=len(pos1))

        # track pairs where j rank is wrong
        weights *= numpy.where(rank_j != comm.rank, 1, 0)
        wrong_rank[:] += numpy.bincount(i, weights=weights, minlength=len(pos1))

        # save the valid pairs for final calculations
        res = numpy.vstack([i, j, sort_i, sort_j]).T
        dataframe.append(res)

    # add all the valid pairs to a dataframe
    tree1.enum(tree2, rmax, process=callback)

    # sorted indices of objects that are centrals
    # (objects with no pairs with j > i)
    centrals = set(sortindex1[(j_gt_i==0)])

    # sorted indices of objects that might be centrals
    # (pairs with j>i that live on other ranks)
    maybes = set(sortindex1[(wrong_rank>0)])

    # store the pairs in a pandas dataframe for fast groupby
    dataframe = numpy.concatenate(dataframe, axis=0)
    df = pd.DataFrame(dataframe, columns=['i', 'j', 'sort_i', 'sort_j'])

    # we sort by the correct sorted index in descending order which puts
    # highest priority objects first
    df.sort_values("sort_i", ascending=False, inplace=True)

    # index by the correct sorted order
    df.set_index('sort_i', inplace=True)

    # to find centrals, considers objects that could be satellites of another
    # (pairs with sort_j > sort_i)
    possible_cens = df[(df['sort_j']>df.index.values)]
    possible_cens = possible_cens.drop(centrals, errors='ignore')
    _remove_objects_paired_with(possible_cens, centrals) # remove objs paired with cens

    # sorted indices of objects that have pairs on other ranks
    # these objects are already "maybe" centrals
    on_other_ranks = sortindex1[(wrong_rank>0)]

    # find the centrals and associated halo labels for each central
    all_centrals, labels = _find_centrals(comm, possible_cens, on_other_ranks, centrals, maybes)

    # reset the index and return
    df.reset_index(inplace=True)

    # add the halo labels for each pair in the dataframe
    labels = pd.Series(labels, name='label_i', index=pd.Index(all_centrals, name='sort_i'))
    df = df.join(labels, on='sort_i')
    labels.name = 'label_j'; labels.index.name = 'sort_j'
    df = df.join(labels, on='sort_j')

    # iniitalize the output arrays
    labels = numpy.zeros(len(pos1), dtype='i8') - 1 # indexed by i
    types = numpy.zeros(len(pos1), dtype='u4') # indexed by i
    counts = numpy.zeros(len(pos2), dtype='i8') # indexed by j

    # assign labels of the centrals
    cens = df.dropna(subset=['label_j']).drop_duplicates('i')
    labels[cens['i'].values] = cens['label_i'].values

    # objects on this rank that are satellites
    # (no label for the 1st object in pair but a label for the 2nd object)
    sats = (df['label_i'].isnull())&(~df['label_j'].isnull())
    df = df[sats]

    # find the corresponding central for each satellite
    df = df.sort_values('sort_j', ascending=False)
    df.set_index('sort_i', inplace=True)
    sats_grouped = df.groupby('sort_i', sort=False, as_index=False)
    centrals = sats_grouped.first() # these are the centrals for each satellite

    # update the satellite info with its pair with the highest priority
    cens_i = centrals['i'].values; cens_j = centrals['j'].values
    counts += numpy.bincount(cens_j, minlength=len(pos2))
    types[cens_i] = 1
    labels[cens_i] = centrals['label_j'].values

    # sum counts across ranks (take the sum of any repeated objects)
    counts = layout2.gather(counts, mode='sum')

    # output fields
    dtype = numpy.dtype([('cgm_haloid', 'i8'),
                         ('num_cgm_sats', 'i8'),
                         ('cgm_type', 'u4'),
                         ('origind', 'u4')])
    out = numpy.empty(len(data), dtype=dtype)

    # gather the data back onto the original ranks
    # no ghosts for this domain layout so choose any particle
    out['cgm_haloid'] = layout1.gather(labels, mode='any')
    out['origind'] = layout1.gather(origind1, mode='any')
    out['num_cgm_sats'] = layout1.gather(counts, mode='any')
    out['cgm_type'] = layout1.gather(types, mode='any')

    # restore the original order
    mpsort.sort(out, orderby='origind', comm=comm)

    fields = ['cgm_type', 'cgm_haloid', 'num_cgm_sats']
    return out[fields]


def _remove_objects_paired_with(df, bad_pairs):
    """
    Remove any objects that are paired with an object in ``bad_pairs``

    This is done in place
    """
    assert df.index.name == 'sort_i'

    df.reset_index(inplace=True)
    df.set_index('sort_j', inplace=True)

    # exception could be raised if no pairs need to be dropped
    # so just reset index and return
    try:
        bad_pair_index = df.index.intersection(bad_pairs).unique()
        to_drop = df.loc[bad_pair_index]['sort_i'].values
        df.reset_index(inplace=True)
        df.set_index('sort_i', inplace=True)
        df.drop(to_drop,inplace=True)
    except:
        df.set_index('sort_i', inplace=True)


def _find_centrals(comm, df, on_other_ranks, centrals, maybes):
    """
    Find the sorted index values of all of the centrals

    Each rank determines local centrals (which have no pairs on
    other ranks), and then root searches the objects spread
    out on multiple ranks

    Returns
    -------
    all_centrals : list
        the sorted index values of all centrals
    labels : list
        corresponding labels; ranging from ``0`` to ``len(all_centrals)``
    """
    def find_local_centrals(grp):

        cenid = grp.index.values[0]

        # group number of the centrals that could host this object
        maybe_grp_nums = grp['sort_j'].values

        # this object is a satellite
        if any(num in centrals for num in maybe_grp_nums):
            return

        # this object could be a satellite
        if any(num in maybes for num in maybe_grp_nums):
            maybes.add(cenid)
            return

        # if we get here, this object is definitely a central
        centrals.add(cenid)

    # only need to examine objects that have all higher priority pairs
    # on the same rank --> if they have pairs on other ranks, then they are already
    # marked as "maybes" centrals
    same_rank_df = df.drop(on_other_ranks, errors='ignore')

    # group by centrals and find the local centrals
    # these are objects with no higher priority pairs
    cens_grouped = same_rank_df.groupby('sort_i', sort=False)
    cens_grouped.apply(find_local_centrals)

    # the pairs associated with objects that might be satellites
    maybe_index = df.index.intersection(list(maybes)).unique()
    maybe_cen_groups = df.loc[maybe_index]

    # gather data on maybes
    maybes_data = comm.gather(maybe_cen_groups)
    all_centrals = numpy.concatenate(comm.allgather(list(centrals)), axis=0)

    # root identifies the remaining centrals
    if comm.rank == 0:

        # keep track of new centrals
        new_centrals = set()
        all_maybes = pd.concat(maybes_data)

        # remove objects paired with a central
        _remove_objects_paired_with(all_maybes, all_centrals)

        # consider objects in sorted order
        all_maybes.sort_index(ascending=False, inplace=True)

        def finalize(grp):
            grp_nums = grp['sort_j'].values
            if not any(num in new_centrals for num in grp_nums):
                new_centrals.add(grp.index.values[0])

        # find out which of the maybes are actually centrals
        maybes_grouped = all_maybes.groupby('sort_i', sort=False)
        maybes_grouped.apply(finalize)

    else:
        new_centrals = None

    # get the list of all centrals on all ranks
    new_centrals = comm.bcast(new_centrals)
    all_centrals = numpy.append(all_centrals, list(new_centrals))

    # sort and create unique halo labels
    all_centrals[::-1].sort()
    labels = numpy.arange(0, len(all_centrals), dtype='i4')

    return all_centrals, labels

def data_to_sort_key(data):
    """
    Convert floating type data to unique integers for sorting
    """
    return numpy.fromstring(data.tobytes(), dtype='u8')
