__all__ = ['notebook']

import matplotlib
import os
_cwd = os.path.split(os.path.abspath(__file__))[0]

for name in __all__:
    path = os.path.join(_cwd, 'notebook.mplstyle')
    globals()[name] = matplotlib.rc_params_from_file(path, use_default_template=False)
