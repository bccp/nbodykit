import numpy
import logging

import kdcount
import pandas as pd
from six import string_types

from nbodykit import CurrentMPIComm
from nbodykit.source.catalog import ArrayCatalog

def data_to_sort_key(data):
    """
    Convert floating type data to unique integers for sorting
    """
    if data.dtype == numpy.float64:
        toret = numpy.fromstring(data.tobytes(), dtype='u8')
    else:
        toret = numpy.fromstring(data.tobytes(), dtype='u4')

    return toret

class CylindricalGroups(object):
    """
    Compute groups of objects using a cylindrical grouping method. We identify
    all satellites within a given cylindrical volume around a central object

    Reference
    ---------
    Okumura, Teppei, et al. "Reconstruction of halo power spectrum from
    redshift-space galaxy distribution: cylinder-grouping method and halo
    exclusion effect", arXiv:1611.04165, 2016.
    """
    logger = logging.getLogger('CylindricalGroups')

    def __init__(self, source, rankby, rperp, rpar, periodic=False, los=None, BoxSize=None):
        """
        Parameters
        ----------
        source : CatalogSource
            the input source of particles providing the 'Position' column; the
            grouping algorithm is run on this catalog
        rperp : float
            the radius of the cylinder in the sky plane (i.e., perpendicular
            to the line-of-sight)
        rpar : float
            the radius along the line-of-sight direction; this is 1/2 the
            height of the cylinder
        rankby : str, list
            a single or list of column names to rank order the input source by
            before computing the cylindrical groups, such that objects ranked first
            are marked as CGM centrals
        periodic : bool; optional
            whether to use periodic boundary conditions
        los : 3-vector, optional
            a unit vector of length 3 providing the line-of-sight direction; if
            set to ``None`` the line-of-sight is determined based on an observer
            based at coordinates (0,0,0). For a simulation box, you can
            specify e.g., [0,0,1] to use the z-axis
        BoxSize : float, 3-vector; optional
            the size of the box of the input data; must be provided if
            ``periodic=True``
        """
        if 'Position' not in source:
            raise ValueError("the 'Position' column must be defined in the input source")

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
        if los is not None:
            if numpy.isscalar(los) or len(los) != 3:
                raise ValueError("line-of-sight ``los`` should be vector with length 3")
            if not numpy.allclose(numpy.einsum('i,i', los, los), 1.0, rtol=1e-5):
                raise ValueError("line-of-sight ``los`` must be a unit vector")

        # save meta-data
        self.attrs['rpar'] = rpar
        self.attrs['rperp'] = rperp
        self.attrs['periodic'] = periodic
        self.attrs['rankby'] = rankby
        self.attrs['los'] = los

        # log some info
        if self.comm.rank == 0:
            args = (str(rperp), str(rpar))
            self.logger.info("finding groups with rperp=%s and rpar=%s " %args)
            if los is None:
                self.logger.info("  using line-of-sight computed using observer at origin (0,0,0)")
            else:
                self.logger.info("  using line-of-sight vector %s" %str(los))
            msg = "periodic" if periodic else "non-periodic"
            self.logger.info("  using %s boundary conditions" %msg)

        self.run()

    def run(self):
        """
        Compute the cylindrical groups, saving the results to the
        :attr:`groups` attribute

        Attributes
        ----------
        groups : ArrayCatalog
            a catalog holding the results of the groups. The length of the
            catalog is equal to the length of the input size, i.e., the length
            is equal to the :attr:`size` attribute. The relevant fields are:

            num_cgm_sats :
                The number of satellites found in each CGM halo
            cgm_cenid :
                The index of the central CGM object in the original catalog;
                if an object is denoted as a CGM satellite, the index here is -1
            cgm_gal_type :
                a flag specifying the type for each object in the original catalog,
                with 0 specifying CGM central and 1 denoting CGM satellite
        """
        from pmesh.domain import GridND
        import mpsort

        comm = self.comm
        if self.attrs['periodic']:
            boxsize = self.attrs['BoxSize']
        else:
            boxsize = None

        # determine processor division for domain decomposition
        for Nx in range(int(comm.size**0.3333) + 1, 0, -1):
            if comm.size % Nx == 0: break
        else:
            Nx = 1
        for Ny in range(int(comm.size**0.5) + 1, 0, -1):
            if (comm.size // Nx) % Ny == 0: break
        else:
            Ny = 1
        Nz = comm.size // Nx // Ny
        Nproc = [Nx, Ny, Nz]
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(Nproc))

        # get the position
        cols = self.source.compute(self.source["Position"], *[self.source[col] for col in self.attrs['rankby']])
        pos = cols[0]
        rankby_cols = list(cols[1:])

        # global min/max across all ranks
        posmin = numpy.asarray(comm.allgather(pos.min(axis=0))).min(axis=0)
        posmax = numpy.asarray(comm.allgather(pos.max(axis=0))).max(axis=0)

        # domain decomposition
        grid = [numpy.linspace(posmin[i], posmax[i], Nproc[i]+1, endpoint=True) for i in range(3)]
        domain = GridND(grid, comm=comm)

        # make the data to sort
        dtype = [('origind', 'u4'),
                 ('sortindex', 'u4'),
                 ('pos', (pos.dtype.str, 3)),
                 ('key', 'u8')]
        for i, name in enumerate(self.attrs['rankby']):
            dtype.append((name, rankby_cols[i].dtype))
        dtype = numpy.dtype(dtype)

        data = numpy.empty(len(pos), dtype=dtype)
        data['pos'] = pos
        for i, name in enumerate(self.attrs['rankby']):
            data[name] = rankby_cols[i]

        # keep track of the original order
        sizes = self.comm.allgather(self.source.size)
        data['origind'] = numpy.arange(self.source.size, dtype='u4')
        data['origind'] += sum(sizes[:self.comm.rank])

        # sort the particles
        for col in self.attrs['rankby'][::-1]:
            dt = data.dtype[col]
            rankby = col
            # make an integer key for floating columns
            if issubclass(dt.type, numpy.floating):
                data['key'] = data_to_sort_key(data[col])
                rankby = 'key'
            elif not issubclass(dt.type, numpy.integer):
                args = (col, str(dt))
                raise ValueError("cannot sort by column '%s' with dtype '%s'; must be integer or floating type" %args)

            # do the sort
            mpsort.sort(data, orderby=rankby, comm=self.comm)

        # keep track of the sorted order
        data['sortindex'] = numpy.arange(self.source.size, dtype='u4')
        data['sortindex'] += sum(sizes[:self.comm.rank])

        # exchange across ranks
        layout1 = domain.decompose(data['pos'], smoothing=0)
        pos1 = layout1.exchange(data['pos'])
        origind1 = layout1.exchange(data['origind'])
        sortindex1 = layout1.exchange(data['sortindex'])
        for i, col in enumerate(rankby_cols):
            rankby_cols[i] = layout1.exchange(col)

        # the maximum distance still inside the cylinder
        rperp, rpar = self.attrs['rperp'], self.attrs['rpar']
        rmax = (rperp**2 + rpar**2)**0.5

        # exchange the positions to correlate against
        layout2  = domain.decompose(data['pos'], smoothing=rmax)
        pos2 = layout2.exchange(data['pos'])
        origind2 = layout2.exchange(data['origind'])
        sortindex2  = layout2.exchange(data['sortindex'])

        # make the KD-trees
        tree1 = kdcount.KDTree(pos1, boxsize=boxsize).root
        tree2 = kdcount.KDTree(pos2, boxsize=boxsize).root

        # square of perp distance
        rperp2 = self.attrs['rperp']**2

        dataframe = []
        def callback(r, i, j):

            r1 = pos1[i]
            r2 = pos2[j]
            dr = r1 - r2

            # enforce periodicity in dpos
            if self.attrs['periodic']:
                for axis, col in enumerate(dr.T):
                    col[col > boxsize[axis]*0.5] -= boxsize[axis]
                    col[col <= -boxsize[axis]*0.5] += boxsize[axis]

            # los distance
            rlos =  numpy.einsum("ij,j->i", dr, self.attrs['los'])

            # sky
            dr2 = numpy.einsum('ij, ij->i', dr, dr)
            rsky2 = numpy.abs(dr2 - rlos ** 2)

            # sort indices
            sorti = sortindex1[i]
            sortj = sortindex2[j]

            # original indices
            origi = origind1[i]
            origj = origind2[j]

            # save the valid pairs
            # NOTE: we sorted in ascending order but actually need descending (most massive first)
            # so we've restricted to to centrals with index greater than satellites
            valid = (sorti>sortj)&(rsky2 <= rperp2)&(abs(rlos) <= self.attrs['rpar'])
            res = numpy.vstack([xx[valid] for xx in [i, j, sorti, sortj, origi, origj]]).T
            dataframe.append(res)

        # enum the tree
        tree1.enum(tree2, rmax, process=callback)

        # make into a pandas DataFrame so we can do fast groupby
        dataframe = numpy.concatenate(dataframe, axis=0)
        df = pd.DataFrame(dataframe, columns=['i', 'j', 'sort_i', 'sort_j', 'orig_i', 'orig_j'])

        # group centrals by index in input data
        # and sort by the sort index
        df = df.sort_values("sort_i", ascending=False)
        cens_grouped = df.groupby('orig_i', sort=False)

        # make the local output result
        Ntot = self.source.csize
        dtype = numpy.dtype([('num_cgm_sats', 'i8'),
                             ('cgm_gal_type', 'i4'),
                             ('cgm_cenid', 'i8'),
                             ('origind', 'u4'),
                             ('local', '?'),
                             ('rank', 'u4')])
        out1 = numpy.empty(Ntot, dtype=dtype)
        out1['num_cgm_sats'] = numpy.zeros(Ntot, dtype='i8')
        out1['cgm_gal_type'] = numpy.zeros(Ntot, dtype='i4') - 1
        out1['cgm_cenid'] = numpy.zeros(Ntot, dtype='i8') - 1
        out1['origind'] = numpy.arange(Ntot, dtype='u4')
        out1['local'] = numpy.zeros(Ntot, dtype='?')
        out1['rank'] = numpy.concatenate([numpy.ones(size, dtype='u4')*i for i, size in enumerate(sizes)])

        # loop over all potential CGM centrals
        for orig_cenid, group in cens_grouped:

            # indices of the all neighbors of this object
            orig_sat_inds = group['orig_j'].values

            # new objects must be unmarked
            if out1['cgm_gal_type'][orig_cenid] == -1:

                # this must be a central
                out1['cgm_gal_type'][orig_cenid] = 0
                out1['local'][orig_cenid] = True

                # only unidentified satellites can be added to this central
                isnew = out1['cgm_gal_type'][orig_sat_inds] == -1
                orig_sat_inds = orig_sat_inds[isnew]

                # update the satellite info
                out1['cgm_gal_type'][orig_sat_inds] = 1
                out1['cgm_cenid'][orig_sat_inds] = orig_cenid
                out1['local'][orig_sat_inds] = True

                # store the number of sats in this halo
                out1['num_cgm_sats'][orig_cenid] = isnew.sum()

        # sort by rank to put data back on the right rank
        mpsort.sort(out1, orderby='rank', comm=self.comm)

        # restrict to local objects that have been computed on this rank
        out1 = out1[out1['local']==True]

        # the ouput groups
        # by default objects are CGM centrals and then we add
        # in those objects with at least one satellite
        dtype = numpy.dtype([('num_cgm_sats', 'i4'),
                             ('cgm_gal_type', 'i4'),
                             ('cgm_cenid', 'i4')])
        groups = numpy.empty(self.source.size, dtype=dtype)
        groups['num_cgm_sats'] = numpy.zeros(len(groups), dtype='i4')
        groups['cgm_gal_type'] = numpy.zeros(len(groups), dtype='i4')
        groups['cgm_cenid'] = numpy.zeros(len(groups), dtype='i4') -  1

        # the local index of the objects with neighbors on this rank
        index = out1['origind'] - sum(sizes[:self.comm.rank])
        for name in groups.dtype.names:
            groups[name][index] = out1[name][:]

        # make the final structured array
        self.groups = ArrayCatalog(groups, comm=self.comm, **self.attrs)

        # log some info
        N_cen = (groups['cgm_gal_type']==0).sum()
        isolated_N_cen = ((groups['cgm_gal_type']==0)&(groups['num_cgm_sats']==0)).sum()
        N_cen = self.comm.allreduce(N_cen)
        isolated_N_cen = self.comm.allreduce(isolated_N_cen)
        if self.comm.rank == 0:
            self.logger.info("found %d CGM centrals total" %N_cen)
            self.logger.info("%d/%d are isolated centrals (no satellites)" % (isolated_N_cen,N_cen))
