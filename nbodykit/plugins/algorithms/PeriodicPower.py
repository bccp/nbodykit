from nbodykit.extensionpoints import Algorithm
import logging

class PeriodicPowerAlgorithm(Algorithm):

    plugin_name = "PeriodicPower"
    
    def __init__(self, mode, Nmesh, output, 
                    los='z', 
                    Nmu=100, 
                    dk=0.005, 
                    kmin=0, 
                    quiet=logging.DEBUG, 
                    poles=[], 
                    pole_output=None,
                    correlation=False):
                    
        self.mode        = mode
        self.Nmesh       = Nmesh
        self.output      = output
        self.los         = los
        self.Nmu         = Nmu
        self.dk          = dk
        self.kmin        = kmin
        self.quiet       = quiet
        self.poles       = poles
        self.pole_output = pole_output
        self.correlation = correlation
        
    @classmethod
    def register(kls):
        
        h = kls.parser
        
        kls.parser.add_argument("mode", choices=["2d", "1d"]) 
        kls.parser.add_argument("Nmesh", type=int, help='size of calculation mesh, recommend 2 * Ngrid')
        kls.parser.add_argument("output", help='write power to this file. set as `-` for stdout') 

        # add the input field types
        #h = "one or two input fields, specified as:\n\n"
        # kls.parser.add_argument("fields", nargs="+",
        #         action=InputAction,
        #         help="Input data sources and painters. Use --list-painter and --list-datasource to see a list of painters and data sources.",
        #         metavar="DataSource [Painter] [DataSource [Painter]]")
        kls.parser.add_argument("--los", choices="xyz", default='z',
                help="the line-of-sight direction, which the angle `mu` is defined with respect to")
        kls.parser.add_argument("--Nmu", type=int, default=5,
                help='the number of mu bins to use; if `mode = 1d`, then `Nmu` is set to 1' )
        kls.parser.add_argument("--dk", type=float,
                help='the spacing of k bins to use; if not provided, the fundamental mode of the box is used')
        kls.parser.add_argument("--kmin", type=float, default=0,
                help='the edge of the first bin to use; default is 0')
        kls.parser.add_argument('-q', '--quiet', help="silence the logging output",
                action="store_const", dest="log_level", const=logging.ERROR, default=logging.DEBUG)
        kls.parser.add_argument('--poles', type=lambda s: [int(i) for i in s.split()], default=[],
                help='if specified, compute these multipoles from P(k,mu), saving to `pole_output`')
        kls.parser.add_argument('--pole_output', type=str, help='the name of the output file for multipoles')

        kls.parser.add_argument("--correlation", action='store_true', default=False,
            help='Calculate correlation function instead of power spectrum.')
        
        
    def run(self):
        raise NotImplementedError
    


