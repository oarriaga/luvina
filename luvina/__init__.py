from __future__ import absolute_import
import os

from . import backend
from . import datasets
# from . import metrics
# from . import models
from . import utils

__version__ = '0.0.9'

# Obtain luvina base dir path: either ~/.luvina or /tmp.
_luvina_base_dir = os.path.expanduser('~')
if not os.access(_luvina_base_dir, os.W_OK):
    _luvina_base_dir = '/tmp'
_luvina_dir = os.path.join(_luvina_base_dir, '.luvina')

if not os.path.exists(_luvina_dir):
    try:
        os.makedirs(_luvina_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


