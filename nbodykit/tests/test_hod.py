from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

setup_logging()

@pytest.mark.parametrize('model', [Zheng07Model, Hearin15Model, Leauthaud11Model])
def test_consistency(model):

    # correct answers
    truth = {}
    truth['cosmology'] = cosmology.Planck15.to_astropy()
    truth['redshift'] = 0.55
    truth['prim_haloprop_key'] = 'halo_m200m'
    truth['halo_boundary_key'] = 'halo_r200m'

    # make the model
    model = model.to_halotools(truth['cosmology'], truth['redshift'], '200m')

    # test consistency of attributes
    attrs = ['cosmology', 'redshift', 'prim_haloprop_key', 'halo_boundary_key']
    for a in attrs:
        vals = []
        for m in model.model_dictionary:
            comp = model.model_dictionary[m]
            y = getattr(comp, a, None)
            if y is not None: vals.append(y)

        assert all(v == truth[a] for v in vals)
