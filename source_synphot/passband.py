# -*- coding: UTF-8 -*-
"""
Instrumental throughput models,  calibration and synthetic photometry
routines
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
from . import io
import numpy as np
import pysynphot as S
from astropy.cosmology import default_cosmology
from collections import OrderedDict

def synflux(spec, pb):
    """
    Compute the synthetic flux of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : :py:class:`pysynphot.ArraySpectrum`
        The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The passband data.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``

    Returns
    -------
    flux : float
        The normalized flux of the spectrum through the passband

    Notes
    -----
        The passband is assumed to be dimensionless photon transmission
        efficiency.

        Routine is intended to be a mch faster implementation of
        :py:meth:`pysynphot.observation.Observation.effstim`, since it is called over and
        over by the samplers as a function of model parameters.

        Uses :py:func:`numpy.trapz` for interpolation.
    """
    overlap = pb.check_overlap(spec)
    if overlap == 'none':
        return np.nan
    elif overlap == 'partial':
        if pb.check_sig(spec):
            pass
        else:
            return np.nan
    else:
        pass
    flux = spec.sample(pb.wave)
    n = np.trapz(flux*pb.wave*pb.throughput, pb.wave)
    d = np.trapz(pb.wave*pb.throughput, pb.wave)
    out = n/d
    return out


def synphot(spec, pb, zp=0.):
    """
    Compute the synthetic magnitude of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : :py:class:`pysynphot.ArraySpectrum`
        The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The passband transmission.
    zp : float, optional
        The zeropoint to apply to the synthetic flux

    Returns
    -------
    mag : float
        The synthetic magnitude of the spectrum through the passband

    See Also
    --------
    :py:func:`source_synphot.passband.synflux`
    """
    flux = synflux(spec, pb)
    m = -2.5*np.log10(flux) + zp
    return m


def syncolor(spec, pb1, pb2, zp1=0., zp2=0.):
    """
    Compute the synthetic color of spectrum ``spec`` between passbands ``pb1``
    and ``pb2``

    Parameters
    ----------
    spec : :py:class:`pysynphot.ArraySpectrum`
        The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pb1 : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        Passband 1 transmission.
    pb2 : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        Passband 2 transmission.
    zp1 : float, optional
        The zeropoint to apply to the synthetic flux through passband ``pb1``
    zp2 : float, optional
        The zeropoint to apply to the synthetic flux through passband ``pb2``

    Returns
    -------
    mag : float
        The synthetic magnitude of the spectrum through the passband

    See Also
    --------
    :py:func:`source_synphot.passband.synflux`
    :py:func:`source_synphot.passband.synphot`
    """
    flux1 = synflux(spec, pb1)
    m1 = -2.5*np.log10(flux1) + zp1
    flux2 = synflux(spec, pb2)
    m2 = -2.5*np.log10(flux2) + zp2
    return m1-m2


def synphot_over_redshifts(spec, redshifts, pb, zp=0.):
    """
    Compute the synthetic magnitude of spectrum ``spec`` through passband ``pb`` over some ``redshifts``

    Parameters
    ----------
    spec : :py:class:`pysynphot.ArraySpectrum`
        The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    redshifts: array-like
        The array of redshifts
    pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The passband transmission.
    zp : float, optional
        The zeropoint to apply to the synthetic flux

    Returns
    -------
    mags : array-like
        The synthetic magnitudes of the spectrum through the passband at
        `redshifts` Output is the same shape as redshifts, with entries where
        the spectra does not overlap the passband set to NaN.

    See Also
    --------
    :py:func:`source_synphot.passband.synflux`
    :py:func:`source_synphot.passband.synphot`
    """
    mags = []
    c = default_cosmology.get()
    mu = c.distmod(redshifts)
    for i, z in enumerate(redshifts):
        if z < 0:
            mags.append(np.nan)
            continue
        this_spec_z = spec.redshift(z)
        this_spec_z.convert('flam')
        mag = synphot(this_spec_z, pb, zp=zp) + mu[i].value
        mags.append(mag)
    mags = np.array(mags)
    return mags


def syncolor_over_redshifts(spec, redshifts, pb1, pb2, zp1=0., zp2=0.):
    """
    Compute the synthetic color of spectrum ``spec`` through passband ``pb`` over some ``redshifts``

    Parameters
    ----------
    spec : :py:class:`pysynphot.ArraySpectrum`
        The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    redshifts: array-like
        The array of redshifts
    pb1 : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The blue passband transmission.
    pb2 : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The red passband transmission.
    zp1 : float, optional
        The zeropoint to apply to the synthetic flux in passband `pb1`
    zp2 : float, optional
        The zeropoint to apply to the synthetic flux in passband `pb2`

    Returns
    -------
    cols : array-like
        The synthetic colors of the spectrum through the passbands at
        `redshifts` Output is the same shape as redshifts, with entries where
        the spectra does not overlap the passbands set to NaN.

    See Also
    --------
    :py:func:`source_synphot.passband.synflux`
    :py:func:`source_synphot.passband.synphot`
    """
    cols = []
    for z in redshifts:
        if z < 0:
            cols.append(np.nan)
            continue
        this_spec_z = spec.redshift(z)
        col = syncolor(this_spec_z, pb1, pb2, zp1=zp1, zp2=zp2)
        cols.append(col)
    cols = np.array(cols)
    return cols


def get_pb_zpt(pb, reference='AB', model_mag=None):
    """
    Determines a passband zeropoint for synthetic photometry, given a reference
    standard and its model magnitude in the passband

    Parameters
    ----------
    pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The passband data.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
    reference : str, optional
        The name of the reference spectrophotometric standard to use to determine the passband zeropoint.
        'AB' or 'Vega' (default: 'AB')
    model_mag : float, optional
        The magnitude of the reference spectrophotometric standard in the passband.
        default = None

    Returns
    -------
    pbzp : float
        passband AB zeropoint

    Raises
    ------
    RuntimeError
        If the passband and reference standard do not overlap
    ValueError
        If the reference model_mag is invalid

    See Also
    --------
    :py:func:`source_synphot.passband.synflux`
    :py:func:`source_synphot.passband.synphot`
    """

    # setup the photometric system by defining the standard and corresponding magnitude system
    if reference.lower() not in ('vega', 'ab'):
        message = 'Do not understand mag system reference standard {}. Must be AB or Vega'.format(reference)
        raise RuntimeError(message)
    if reference.lower() == 'ab':
        refspec = S.FlatSpectrum(3631, fluxunits='jy')
        mag_type= 'abmag'
    else:
        refspec    = S.Vega
        mag_type= 'vegamag'

    refspec.convert('flam')

    if model_mag is None:
        ob = S.Observation(refspec, pb)
        model_mag = ob.effstim(mag_type)

    synphot_mag = synphot(refspec, pb, zp=0.)
    outzp = model_mag - synphot_mag
    return outzp


def load_pbs(pbnames, model_mags, model='AB'):
    """
    Loads passbands, and calibrates their zeropoints so that ``model`` has
    magnitude ``model_mags`` through them.

    Parameters
    ----------
    pbnames : array-like
        The passband names. Passed to :py:func:`source_synphot.io.read_passband`
    model_mags : array-like
        The magnitudes of ``model`` in the passbands ``pbnames``
    model : str, optional
        The reference model for the passband. Either ``'AB'`` or ``'Vega'``.
        The same reference model is used for all the passbands. If the
        passbands have different standards they are calibrated to, then call
        the function twice, and concatenate the output.

    Returns
    -------
    pbs : dict
        The dictionary of passband transmissions and zeropoints, such that
        :py:func:`synphot` of ``model`` through passband ``pbnames`` returns
        magnitude ``model_mags``.

    Raises
    ------
    ValueError
        If the number of ``model_mags`` does not match the number of passbands
        in ``pbnames``

    Notes
    -----
        The passband zeropoint computed here is what number must be used with
        :py:func:`synphot` and ``model``. It is not what observers think of as
        the zeropoint, which invariable encapsulates the telescope collecting
        area.

    See Also
    --------
    :py:func:`source_synphot.io.read_passband`
    :py:func:`source_synphot.passband.get_pb_zpt`
    """

    pbs = OrderedDict()
    if np.isscalar(pbnames):
        pbnames = np.array(pbnames, ndmin=1)
    else:
        pbnames = np.array(pbnames).flatten()
    npb = len(pbnames)

    if np.isscalar(model_mags):
        model_mags = np.repeat(model_mags, npb)
    else:
        model_mags = np.array(model_mags).flatten()

    if len(model_mags) != npb:
        message = 'Model mags and pbnames must be 1-D arrays with the same shape'
        raise ValueError(message)

    for i, pbmag in enumerate(zip(pbnames, model_mags)):
        pb, mag = pbmag
        try:
            thispb, _ = io.read_passband(pb)
        except Exception as e:
            print(e)
            message = 'Passband {} not loaded'.format(pb)
            warnings.warn(message, RuntimeWarning)
            continue
        thispbzp = get_pb_zpt(thispb, model_mag=mag, reference=model)
        pbs[pb] = thispb, thispbzp

    return pbs
