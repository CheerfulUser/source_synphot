import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,'/Users/rridden/Documents/work/code/source_synphot/'))
import source_synphot.passband as passband
import source_synphot.io as io
import source_synphot.source
import astropy.table as at
from collections import OrderedDict
import pysynphot as S
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import astropy.table as at
from extinction import fm07, fitzpatrick99, apply, remove


import pandas as pd
from glob import glob


def mangle(spec,pbs,mags,plot=False):
    """
    Mangle a spectrum to scale it to observed photometry.

    Inputs
    ------
        spec : pysynphot spectrum array

        pbs : ordered dict 
            dictionary of passbands. first entry is a pysynphot array passband second is the band zeropoint.
        mags : list/array
            list of magnitudes in the same order as the bands
    Options
    -------
        plot : bool
            if true, plots the scale factor

    Returns
    -------
        s : pysynphot spectrum array
            input spectrum scalled by the mangling process
    """
    scale = np.array((spec.wave,spec.flux))
    scale[1,:] = 1
    i = 0
    inds = []
    for pb in pbs:
        filt = pbs[pb]
        syn_mag = source_synphot.passband.synphot(spec,filt[0],zp=filt[1])
        factor = 10**(-2/5*(mags[i]-syn_mag))
        med_wav = np.average(filt[0].wave,weights = filt[0].throughput)
        ind = np.argmin(abs(scale[0,:] - med_wav))
        inds += [ind]
        scale[1,ind] = factor
        i += 1 
    inds.sort()
    # Scipy interpolation, more flexibility in fit
    #interp = interp1d(scale[0,inds],scale[1,inds],kind='linear',bounds_error=False,fill_value=0)
    #interped = interp(scale[0,:])
    #interped[:min(inds)] = scale[1,min(inds)]
    #interped[:max(inds)] = scale[1,max(inds)]
    
    factors = np.interp(scale[0,:],scale[0,inds],scale[1,inds])
    scale[1,:] = factors
    s = S.ArraySpectrum(spec.wave,spec.flux*scale[1,:])
    if plot:
        plt.figure()
        plt.plot(scale[0,:],factors,'.',label='Spline')
        plt.plot(scale[0,inds],factors[inds],'x',label='references')
        plt.plot(spec.wave,spec.flux/np.nanmax(spec.flux),label='original')
        plt.plot(s.wave,s.flux/np.nanmax(s.flux),label='mangled')
        plt.xlabel('Wave')
        plt.ylabel('normed flux/scale factor')
        plt.savefig('mangle{}.png'.format(spec))
        print('mangle{}.png'.format(spec))
    return s


def Spec_mags(Models,pbs,av=0,Rv=3.1,Conversion = 1.029):
    """
    Generate synthetic magnitudes from the models and passbands added.
    Conversion converts between Ebv and Egr, the Green value is 0.981, but the best fit value
    was found to be 1.029.
    """
    #a_v = 3.1*(Conversion * ex ) # ex = extinction from Bayestar19 = Egr
    keys = list(pbs.keys())
    mags = {}
    for key in keys:
        mags[key] = []
    
        pb, zp = pbs[key]
    
        # construct mags
        ind = []
        red = {}
        for model in Models:
            if av > 0:
                model = S.ArraySpectrum(model.wave,apply(fitzpatrick99(model.wave,av,Rv),model.flux),
                                         waveunits=model.waveunits,fluxunits=model.fluxunits)
            if av < 0:
                model = S.ArraySpectrum(model.wave,remove(fitzpatrick99(model.wave,-av,Rv),model.flux),
                                         waveunits=model.waveunits,fluxunits=model.fluxunits)
            mags[key] += [source_synphot.passband.synphot(model, pb,zp)]

    for key in keys:
        mags[key] = np.array(mags[key])
        
    #good = np.ones(len(mags[key])) > 0
    #for key in keys:
    #    good = good *np.isfinite(mags[key])
    #for key in keys:
    #    mags[key] = mags[key][good]
    return mags


def Specs(Specs):
    specs = {}
    for spec in Specs:
        print(spec)
        model_sed = source_synphot.source.pre_process_source(spec,np.nan,'ps1g',0,Renorm=False)
        specs[spec] = model_sed

    return specs


def Syn_mag(pbs,spec):
    mag = {}
    for pb in pbs:
        if spec is not None:
            syn_mag = source_synphot.passband.synphot(spec,pbs[pb][0],zp=pbs[pb][1])
        else:
            syn_mag = np.nan
        mag[pb] = syn_mag
        
    return mag