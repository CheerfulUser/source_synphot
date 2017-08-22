# -*- coding: UTF-8 -*-
"""
Source processing routines
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
from collections import OrderedDict
import numpy as np
import os
import pysynphot as S
import astropy.table as at
from . import io
from . import passband


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


def pre_process_source(source, sourcemag, sourcepb, sourcez, smooth=True):
    """
    Pre-process a source at some redshift ``sourcez`` back to the rest-frame
    and normalize it to have magnitude ``sourcemag`` in passband ``sourcepb``

    Parameters
    ----------
    sourcespec : str
            The source spectrum filename
    sourcemag : float
            The magnitude of the source spectrum in passband ``sourcepb``
    sourcepb : :py:class:`pysynphot.spectrum.ArraySpectralElement`
            The passband in which `source` has magnitude ``sourcemag``
    sourcez : float
            The redshift of `source`
    smooth : bool, optional
            Smooth the spectrum (default: True)

    Returns
    -------
    source : :py:class:`pysynphot.ArraySpectrum`
        The de-redshifted, normalized and optionally smoothed spectrum

    See Also
    --------
    :py:func:`astropy.table.Table.read`
    """
    inspec    = None
    inspecz   = np.nan
    inspecmag = np.nan
    inspecpb  = None

    source_table_file = os.path.join('sources', 'sourcetable.txt')
    source_table_file = io.get_pkgfile(source_table_file)
    source_table = at.Table.read(source_table_file, format='ascii')
    ind = (source_table['specname'] == source)
    nmatch = len(source_table['specname'][ind])
    if nmatch == 1:
        # load the file and the info
        inspec    = source_table['specname'][ind][0]
        inspecz   = source_table['redshift'][ind][0]
        inspecmag = source_table['g'][ind][0] # for now, just normalize the g-band mag
    elif nmatch == 0:
        message = 'Spectrum {} not listed in lookup table'.format(source)
        pass
    else:
        message = 'Spectrum {} not uniquely listed in lookup table'.format(source)
        pass

    if inspec is None:
        warnings.warn(message, RuntimeWarning)
        inspec    = source
        inspecz   = sourcez
        inspecmag = sourcemag
        inspecpb  = sourcepb

    if not os.path.exists(inspec):
        message = 'Spectrum {} could not be found'.format(inspec)
        raise ValueError(message)

    try:
        spec = at.Table.read(inspec, names=('wave','flux'), format='ascii')
    except Exception as e:
        message = 'Could not read file {}'.format(source)
        raise ValueError(message)

    if hasattr(inspecpb,'wave') and hasattr(inspecpb, 'throughput'):
        pass
    else:
        pbs = passband.load_pbs([inspecpb], 0.)
        try:
            inspecpb = pbs[inspecpb][0]
        except KeyError as e:
            message = 'Could not load passband {}'.format(inspecpb)
            raise RuntimeError(message)

    try:
        inspecmag = float(inspecmag)
    except (TypeError, ValueError) as e:
        message = 'Source magnitude {} could not be interpreted as a float'.format(inspecmag)
        raise ValueError(message)

    try:
        inspecz = float(inspecz)
    except (TypeError, ValueError) as e:
        message = 'Source redshift  {} could not be interpreted as a float'.format(inspecz)
        raise ValueError(message)

    if inspecz < 0 :
        message = 'Source must have positive definite cosmological redshift'
        raise ValueError(message)

    inspec = S.ArraySpectrum(spec['wave'], spec['flux'], fluxunits='flam')
    zblue = 1./(1+inspecz) - 1.
    inspec_rest = inspec.redshift(zblue)
    # TODO renorm is basic and just calculates dmag = RNval - what the original spectrum's mag is
    # and renormalizes - there's some sanity checking for overlaps
    # we can do this without using it and relying on the .passband routines
    try:
        out = inspec_rest.renorm(sourcemag, 'ABmag', inspecpb)
    except Exception as e:
        message = 'Could not renormalize spectrum {}'.format(inspec)
        raise RuntimeError(message)
    return out

