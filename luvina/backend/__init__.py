from __future__ import absolute_import
from __future__ import print_function
import sys

_BACKEND = 'nltk'
if _BACKEND == 'nltk':
    sys.stderr.write('Using NLTK backend.\n')
    from .nltk_backend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND))


def backend():
    """Public method for determining
    the current backend.
    """
    return _BACKEND
