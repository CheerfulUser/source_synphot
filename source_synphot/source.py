# -*- coding: UTF-8 -*-
"""
Source processing routines
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
from . import io
import numpy as np
import pysynphot as S
from collections import OrderedDict


def load_source(sourcenames):
    """
    Loads sources

    Parameters
    ----------
    sourcenames : array-like
        The source names. Passed to :py:func:`source_synphot.io.read_source`

    Returns
    -------
    sources : dict
        The dictionary of source spectra

    See Also
    --------
    :py:func:`source_synphot.io.read_source`
    """

    sources = OrderedDict()
    if np.isscalar(sourcenames):
        sourcenames = np.array(sourcenames, ndmin=1)
    else:
        sourcenames = np.array(sourcenames).flatten()
    nsource = len(sourcenames)

    for source in sourcenames:
        try:
            thissource  = io.read_source(source)
        except Exception as e:
            message = 'Source {} not loaded'.format(source)
            warnings.warn(message, RuntimeWarning)
            continue
        sources[source] = thissource

    return sources
