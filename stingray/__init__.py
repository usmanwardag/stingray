import sys

if sys.version_info.major == 2:
    range = xrange
elif sys.version_info.major == 3:
    range = range
else:
    raise Exception("Python version not supported!")

from stingray.lightcurve import *
from stingray.utils import *
from stingray.powerspectrum import *
from stingray.parametricmodels import *
from stingray.posterior import *
from stingray.parameterestimation import *
