from nbodykit.core import Algorithm, DataSource
import numpy
from kdcount import cluster

class FOF6DAlgorithm(Algorithm):
    """
    An algorithm to find subhalos from FOF groups; a variant of FOF6D
    """
    plugin_name = "FOF6D"

    def __init__(self, datasource, halolabel, linklength=0.078, vfactor=0.368, nmin=32):
        
        # set the input parameters
        self.datasource = datasource
        self.halolabel  = halolabel
        self.linklength = linklength
        self.vfactor    = vfactor
        self.nmin       = nmin
        

    @classmethod
    def fill_schema(cls):

        s = cls.schema
        s.description = "finding subhalos from FOF groups; a variant of FOF6D"
        
        s.add_argument("datasource", type=DataSource.from_config,
            help='`DataSource` objects to run FOF against; ' 
                 'run `nbkit.py --list-datasources` for all options')
        s.add_argument("halolabel", type=DataSource.from_config,
            help='data source for the halo label files; column name is Label')
        s.add_argument("linklength", type=float, help='the linking length')
        s.add_argument("vfactor", type=float,
               help='velocity linking length in units of 1d velocity dispersion.')
        s.add_argument("nmin", type=int, help='minimum number of particles in a halo')

    def run(self):
        """
        Run the FOF6D Algorithm
        """
        import mpsort
        from mpi4py import MPI
        
        comm = self.comm
        offset = 0
        
        with self.halolabel.open() as stream:
            [[Label]] = stream.read(['Label'], full=True)
        mask = Label != 0
        dtype = numpy.dtype([
                ('Position', ('f4', 3)), 
                ('Velocity', ('f4', 3)), 
                ('Label', ('i4')), 
                ('Rank', ('i4')), 
                ])
        PIG = numpy.empty(mask.sum(), dtype=dtype)
        PIG['Label'] = Label[mask]
        del Label
        with self.datasource.open() as stream:
            [[Position]] = stream.read(['Position'], full=True)
            PIG['Position'] = Position[mask]
            del Position
            [[Velocity]] = stream.read(['Velocity'], full=True)
            PIG['Velocity'] = Velocity[mask]
            del Velocity
     
        Ntot = comm.allreduce(len(mask))
        del mask

        Nhalo = comm.allreduce(
            PIG['Label'].max() if len(PIG['Label']) > 0 else 0, op=MPI.MAX) + 1

        # now count number of particles per halo
        PIG['Rank'] = PIG['Label'] % comm.size

        Nlocal = comm.allreduce(
                    numpy.bincount(PIG['Rank'], minlength=comm.size)
                 )[comm.rank]

        PIG2 = numpy.empty(Nlocal, PIG.dtype)

        mpsort.sort(PIG, orderby='Rank', out=PIG2, comm=self.comm)
        del PIG

        assert (PIG2['Rank'] == comm.rank).all()

        PIG2.sort(order=['Label'])

        self.logger.info('halos = %d', Nhalo)
        cat = []
        for haloid in numpy.unique(PIG2['Label']):
            hstart = PIG2['Label'].searchsorted(haloid, side='left')
            hend = PIG2['Label'].searchsorted(haloid, side='right')
            if hend - hstart < self.nmin: continue
            assert(PIG2['Label'][hstart:hend] == haloid).all()
            cat.append(
                subfof(
                    PIG2['Position'][hstart:hend], 
                    PIG2['Velocity'][hstart:hend], 
                    self.linklength * (self.datasource.BoxSize.prod() / Ntot) ** 0.3333, 
                    self.vfactor, haloid, Ntot, self.datasource.BoxSize))
        cat = numpy.concatenate(cat, axis=0)
        return cat, Ntot

    def save(self, output, data):
        """
        Save the result
        """
        import h5py
        
        comm = self.comm
        cat, Ntot = data
        cat = comm.gather(cat)

        if comm.rank == 0:
            cat = numpy.concatenate(cat, axis=0)
            with h5py.File(output, mode='w') as f:
                dataset = f.create_dataset('Subhalos', data=cat)
                dataset.attrs['LinkingLength'] = self.linklength
                dataset.attrs['VFactor'] = self.vfactor
                dataset.attrs['Ntot'] = Ntot
                dataset.attrs['BoxSize'] = self.datasource.BoxSize

def subfof(pos, vel, ll, vfactor, haloid, Ntot, boxsize):
    nbar = Ntot / boxsize.prod()
    first = pos[0].copy()
    pos -= first
    pos /= boxsize
    pos[pos > 0.5]  -= 1.0 
    pos[pos < -0.5] += 1.0 
    pos *= boxsize
    pos += first

    oldvel = vel.copy()
    vmean = vel.mean(axis=0, dtype='f8')
    vel -= vmean
    sigma_1d = (vel** 2).mean(dtype='f8') ** 0.5
    vel /= (vfactor * sigma_1d)
    vel *= ll
    data = numpy.concatenate(( pos, vel), axis=1)
    #data = pos

    data = cluster.dataset(data)
    Nsub = 0
    thresh = 80
    fof = cluster.fof(data, linking_length=ll, np=0)

    while Nsub == 0 and thresh > 1:
        # reducing the threshold till we find something..
        Nsub = (fof.length > thresh).sum()
        thresh *= 0.9
    # if nothing is found then assume this FOF group is a fluke.
 
    dtype = numpy.dtype([
        ('Position', ('f4', 3)),
        ('Velocity', ('f4', 3)),
        ('LinkingLength', 'f4'),
        ('R200', 'f4'),
        ('R500', 'f4'),
        ('R1200', 'f4'),
        ('R2400', 'f4'),
        ('R6000', 'f4'),
        ('Length', 'i4'),
        ('HaloID', 'i4'),
        ])
    output = numpy.empty(Nsub, dtype=dtype)

    output['Position'][...] = fof.center()[:Nsub, :3]
    output['Length'][...] = fof.length[:Nsub]
    output['HaloID'][...] = haloid
    output['LinkingLength'][...] = ll

    for i in range(3):
        output['Velocity'][..., i] = fof.sum(oldvel[:, i])[:Nsub] / output['Length']

    del fof
    del data
    data = cluster.dataset(pos)

    for i in range(Nsub):
        center = output['Position'][i] 
        rmax = (((pos - center) ** 2).sum(axis=-1) ** 0.5).max()
        r1 = rmax
        output['R200'][i] = so(center, data, r1, nbar, 200.)
        output['R500'][i] = so(center, data, r1, nbar, 500.)
        output['R1200'][i] = so(center, data, output['R200'][i] * 0.5, nbar, 1200.)
        output['R2400'][i] = so(center, data, output['R1200'][i] * 0.5, nbar, 2400.)
        output['R6000'][i] = so(center, data, output['R2400'][i] * 0.5, nbar, 6000.)
    output.sort(order=['Length', 'Position'])
    print(output)
    output = output[::-1]
    return output

def so(center, data, r1, nbar, thresh=200):
    center = numpy.array([center])
    dcenter = cluster.dataset(center)

    def delta(r):
        if r < 1e-7:
            raise StopIteration
        N = data.tree.root.count(dcenter.tree.root, r)
        n = N / (4. / 3. * numpy.pi * r ** 3)
        return 1.0 * n / nbar - 1
     
    try:
        d1 = delta(r1)
        while d1 > thresh:
            r1 *= 1.4
            d1 = delta(r1)
        # d1 < 200
        r2 = r1
        d2 = d1
        while d2 < thresh:
            r2 *= 0.7
            d2 = delta(r2)
        # d2 > 200

        while True:
            r = (r1 * r2) ** 0.5
            d = delta(r)
            x = (d - thresh)
            if x > 0.1:
                r2 = r
            elif x < -0.1:
                r1 = r
            else:
                return r

    except StopIteration:
        return numpy.nan
