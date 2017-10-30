from nbodykit.lab import *
import argparse

def main(cpus_per_task):

    # the bias values to iterate over
    biases = [1.0, 2.0, 3.0, 4.0]

    # initialize the task manager to run the tasks
    with TaskManager(cpus_per_task=2, use_all_cpus=True) as tm:

    # set up the linear power spectrum
    redshift = 0.55
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')

    # iterate through the bias values
    for bias in tm.iterate(biases):

      # initialize the catalog for this bias
      cat = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=1380., Nmesh=256, bias=bias)

      # compute the power spectrum
      r = FFTPower(cat, mode="2d")

      # and save
      r.save("power-" + str(bias) + ".json")

if __name__ == '__main__':

    desc = "an nbodykit example script using the TaskManager class"
    parser = argparse.ArgumentParser(description=desc)

    h = 'the number of cpus per task'
    parser.add_argument('cpus_per_task', type=int, help=h)

    ns = parser.parse_args()
    main(ns.cpus_per_task)
