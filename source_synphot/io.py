# -*- coding: UTF-8 -*-
"""
I/O methods. All the submodules of the source_synphot package use this module for
almost all I/O operations.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
import argparse
import astropy.table as at
import pysynphot as S

def get_options(args=None):
    """
    Get command line options
    
    Parameters
    ----------
    args : None or array-like
        list of the input command line arguments, typically from
        :py:data:`sys.argv` (which is used if None)

    Returns
    -------
    args : Namespace
        Parsed command line options

    Raises
    ------
    ValueError
        If any input value is invalid

    """
    from argparse import ArgumentParser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                    description=__doc__,\
                    epilog="source_synphot always assumes passbands are %\
                    photon transmission. If your passbands are defined as %\
                    energy transmission at each wavelength, convert them.\
                    Redshifts are assumed to be in the CMB frame.")

    source_input = parser.add_argument_group('source input', 'Source Input Options')
    source_input.add_argument('--sourcespec', required=True,\
            help="Specify source spectrum - relative F_lam")
    source_input.add_argument('--sourcez', required=False, type=float, default=0.,\
            help='Specify redshift of source spectrum')
    source_input.add_argument('--sourcepb', required=False,\
            help="Specify source passband")
    source_input.add_argument('--sourcemag', required=False, type=float, default=0.,\
            help="Specify source magnitude in passband")
    source_input.add_argument('--sourcepbzp', required=False, type=float, default=0.,\
            help="Specify zeropoint of source passband (default uses lookup table)")

    target_output = parser.add_argument_group('target output', 'Target Output Options')
    target_output.add_argument('--targetz', required=False, type=float, default=0.,\
            help='Specify redshift of target')
    target_output.add_argument('--targetpbs', required=False, nargs='+',\
            help="Specify target output passband(s) - default will use all in lookup table")

    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args=args)

    if args.sourcez < 0:
        message = 'Source CMB frame redshift cannot be less than 0'
        raise ValueError(message)

    if args.targetz < 0:
        message = 'Target CMB frame redshift cannot be less than 0'
        raise ValueError(message)

    return args
    

def get_source(sourcespec):
    """
    Read a spectrum 
    
    Parameters
    ----------
    sourcespec : str
        Filename of the ASCII file. Must have columns ``wave``, ``flux``,
        ``flux_err`` or source spectrum specification string. Acceptable
        options are:
         * BB<Temperature>
         * PL<Reference Wave Angstrom>_<PL Index>
         * Flat<AB magnitude>
    Returns
    -------
    source : :py:class:`numpy.recarray`
        Record array with the spectrum data. 
        Has ``dtype=[('wave', '<f8'), ('flux', '<f8'), ('flux_err', '<f8')]``
        `flux_err` is set to 0. for pre-defined sources.

    See Also
    --------
    :py:func:`astropy.table.Table.read`
    :py:func:`numpy.genfromtxt`
    :py:func:`_read_ascii`
    """
    if os.path.exists(sourcespec):
        spec = at.Table.read(sourcespec, names=('wave','flux','dflux'))
    else:
        # assume sourcespec is a string and parse it
        if sourcespec.startswith('BB'):
            temp = sourcespec.lstrip('BB')
            try:
                temp = float(temp)
            except (TypeError, ValueError) as e:
                message = 'Source temperature {} cannot be interpreted as a float'.format(sourcespec)
                raise ValueError(message)
            bb = S.Blackbody(temp)
            spec = {'wave':bb.wave, 'flux':bb.flux, 'dflux':np.repeat(0., len(bb.wave))}
        elif sourcespec.startswith('PL'):
            refwave, plindex = sourcespec.lstrip('PL').split('_')
            try:
                refwave = float(refwave)
            except (TypeError, ValueError) as e:
                message = 'Reference wavelength {} cannot be interpreted as a float'.format(sourcespec)
                raise ValueError(message)
            try:
                plindex = float(plindex)
            except (TypeError, ValueError) as e:
                message = 'Power law index {} cannot be interpreted as a float'.format(sourcespec)
                raise ValueError(message)
            pl = S.PowerLaw(refwave, plindex)
            spec = {'wave':pl.wave, 'flux':pl.flux, 'dflux':np.repeat(0., len(pl.wave))}
        elif sourcespec.startswith('Flat'):
            abmag = sourcespec.replace('Flat','')
            try:
                abmag = float(abmag)
            except (TypeError, ValueError) as e:
                message = 'AB mag {} cannot be interpreted as a float'.format(sourcespec)
                raise ValueError(message)
            flat = S.FlatSpectrum(abmag, fluxunits='abmag')
            spec = {'wave':flat.wave, 'flux':flat.flux, 'dflux':np.repeat(0., len(flat.wave))}
        else:
            message = 'Source spectrum {} cannot be parsed as input file or pre-defined type (BB, PL, Flat)'.format(sourcespec)
            raise ValueError(message)
    spec = np.rec.fromarrays((spec['wave'], spec['flux'], spec['dflux']), names='wave,flux,dflux')
    return spec

