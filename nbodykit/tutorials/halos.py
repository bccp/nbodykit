from nbodykit.source.catalog import HaloCatalog, HDFCatalog
from nbodykit import CurrentMPIComm, transform
from nbodykit.cosmology import Cosmology

class DemoHaloCatalog(HaloCatalog):
    """
    Create a demo catalog of halos using one of the built-in :mod:`halotools`
    catalogs.

    .. note::
        The first request for a particular catalog will download the data
        and cache in the ``~/.astropy/cache/halotools`` directory.

    Parameters
    ----------
    simname : string
        Nickname of the simulation. Currently supported simulations are
        Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
        MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).
    halo_finder : string
        Nickname of the halo-finder, e.g. ``rockstar`` or ``bdm``.
    redshift : float
        Redshift of the requested snapshot.
        Must match one of the available snapshots within ``dz_tol=0.1``,
        or a prompt will be issued providing the nearest
        available snapshots to choose from.

    Examples
    --------
    >>> from nbodykit.tutorials import DemoHaloCatalog
    >>> halos = DemoHaloCatalog('bolshoi', 'rockstar', 0.5)
    >>> print(halos.columns)
    """
    @CurrentMPIComm.enable
    def __init__(self, simname, halo_finder, redshift, comm=None):

        from halotools.sim_manager import CachedHaloCatalog, DownloadManager
        from halotools.sim_manager.supported_sims import supported_sim_dict

        # do seme setup
        self.comm = comm
        meta_cols = ['Lbox', 'redshift', 'particle_mass']

        # try to automatically load from the Halotools cache
        exception = None
        if self.comm.rank == 0:
            kws = {'simname':simname, 'halo_finder':halo_finder, 'redshift':redshift}
            try:
                cached_halos = CachedHaloCatalog(dz_tol=0.1, **kws)
                fname = cached_halos.fname # the filename to load
                meta = {k:getattr(cached_halos, k) for k in meta_cols}
            except Exception as e:

                # try to download on the root rank
                try:
                    # download
                    dl = DownloadManager()
                    dl.download_processed_halo_table(dz_tol=0.1, **kws)

                    # access the cached halo catalog and get fname attribute
                    # NOTE: this does not read the data
                    cached_halos = CachedHaloCatalog(dz_tol=0.1, **kws)
                    fname = cached_halos.fname
                    meta = {k:getattr(cached_halos, k) for k in meta_cols}
                except Exception as e:
                    exception = e
        else:
            fname = None
            meta = None

        # re-raise a download error on all ranks if it occurred
        exception = self.comm.bcast(exception, root=0)
        if exception is not None:
            raise exception

        # broadcast the file we are loading
        fname = self.comm.bcast(fname, root=0)
        meta = self.comm.bcast(meta, root=0)

        # initialize an HDF catalog and add Position/Velocity
        cat = HDFCatalog(fname, comm=comm)
        cat['Position'] = transform.StackColumns(cat['halo_x'], cat['halo_y'], cat['halo_z'])
        cat['Velocity'] = transform.StackColumns(cat['halo_vx'], cat['halo_vy'], cat['halo_vz'])

        # get the cosmology from Halotools
        cosmo = supported_sim_dict[simname]().cosmology # this is astropy cosmology
        cosmo = Cosmology.from_astropy(cosmo)

        # initialize the HaloCatalog
        HaloCatalog.__init__(self, cat, cosmo, meta['redshift'], mdef='vir', mass='halo_mvir')

        # add some meta-data
        # NOTE: all Halotools catalogs have to these attributes
        self.attrs['BoxSize'] = meta['Lbox']
        self.attrs['redshift'] = meta['redshift']
        self.attrs['particle_mass'] = meta['particle_mass']

        # save the cosmology
        self.cosmo = cosmo
        self.attrs['cosmo'] = dict(self.cosmo)
