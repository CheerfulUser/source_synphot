import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,'../source_synphot/'))
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

import pandas as pd
from glob import glob

from scipy.optimize import minimize
from astropy.stats import sigma_clip

from scipy.interpolate import UnivariateSpline


def myround(x, prec=2, base=.5):
    return round(base * round(float(x)/base),prec)

import sys
sys.path.append(os.path.join(dirname,'../Sigma_clip/'))
import sigmacut

import warnings
warnings.filterwarnings("ignore")
from extinction import fm07, fitzpatrick99, apply

import os

def Save_space(Save):
    """
    Creates a pathm if it doesn't already exist.
    """
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        pass


def Tonry_clip(Colours):
    """
    Use the Tonry 2012 PS1 splines to sigma clip the observed data.
    """
    tonry = np.loadtxt(os.path.join(dirname,'Tonry_splines.txt'))
    X = 'r-i'
    Y = 'g-r'
    x = Colours['obs r-i'][0,:]
    mx = tonry[:,0]
    y = Colours['obs g-r'][0,:]
    my = tonry[:,1]
    # set up distance matrix
    xx = x[:,np.newaxis] - mx[np.newaxis,:]
    yy = y[:,np.newaxis] - my[np.newaxis,:]
    # calculate distance
    dd = np.sqrt(xx**2 + yy**2)
    # return min values for the observation axis
    mins = np.nanmin(dd,axis=1)
    # Sigma clip the distance data
    ind = np.isfinite(mins)
    sig = sigma_mask(mins[ind])
    # return sigma clipped mask
    ind[ind] = ~sig
    return ind

def Tonry_residual(Colours):
    """
    Calculate the residuals of the observed data from the Tonry et al 2012 PS1 splines.
    """
    tonry = np.loadtxt(os.path.join(dirname,'Tonry_splines.txt'))
    X = 'r-i'
    Y = 'g-r'
    x = Colours['obs ' + X][0,:]
    mx = tonry[:,0]
    y = Colours['obs ' + Y][0,:]
    my = tonry[:,1]
    # set up distance matrix
    xx = x[:,np.newaxis] - mx[np.newaxis,:]
    yy = y[:,np.newaxis] - my[np.newaxis,:]
    # calculate distance
    dd = np.sqrt(xx**2 + yy**2)
    # return min values for the observation axis
    mingr = np.nanmin(dd,axis=1)
    return np.nansum(mingr) #+ np.nansum(miniz)

def Tonry_fit(K,Data,Model,Compare):
    """
    Wrapper for the residuals function
    """
    Colours = Make_colours(Data,Model,Compare,Extinction = K,Redden=False, Tonry = True)
    res = Tonry_residual(Colours)
    return res

def Tonry_reduce(Data,plot=False):
    '''
    Uses the Tonry et al. 2012 PS1 splines to fit dust and find all outliers.
    '''
    data = copy.deepcopy(Data)
    tonry = np.loadtxt(os.path.join(dirname,'Tonry_splines.txt'))
    compare = np.array([['r-i','g-r']])   
    #cind =  ((data['campaign'].values == Camp))
    dat = data#.iloc[cind]
    clips = []
    if len(dat) < 10:
        raise ValueError('No data available')
    for i in range(2):
        if i == 0:
            k0 = 0.01
        else:
            k0 = res.x

        bds = [(0,10)]
        res = minimize(Tonry_fit,k0,args=(dat,tonry,compare),bounds=bds)#method='Nelder-Mead')
        #res = minimize(Tonry_fit,k0,args=(dat,tonry,compare),method='Nelder-Mead')
        
        colours = Make_colours(dat,tonry,compare,Extinction = res.x, Tonry = True)
        clip = Tonry_clip(colours)
        clips += [clip]
        dat = dat.iloc[clip]
        #print('Pass ' + str(i+1) + ': '  + str(res.x[0]))
    clips[0][clips[0]] = clips[1]
    if plot:
        #print(max(dat['rMeanPSFMag'] - dat['iMeanPSFMag']))
        colours = Make_colours(dat,tonry,compare,Extinction = res.x, Tonry = True)
        plt.figure()
        plt.plot(colours['obs r-i'][0],colours['obs g-r'][0],'.')
        plt.plot(colours['mod r-i'],colours['mod g-r'])
        #print(np.nanmax(colours['obs r-i']))
        plt.xlabel('$r-i$')
        plt.ylabel('$g-r$')
    #clipped_data = data.iloc[clips[0]] 
    return res.x, dat



def Spec_mags(Models,pbs,ex=0,Conversion = 1.029):
    """
    Generate synthetic magnitudes from the models and passbands added.
    Conversion converts between Ebv and Egr, the Green value is 0.981, but the best fit value
    was found to be 1.029.
    """
    a_v = 3.1*(Conversion * ex ) # ex = extinction from Bayestar19 = Egr
    pbg, zpg = pbs['ps1g']
    pbr, zpr = pbs['ps1r']
    pbi, zpi = pbs['ps1i']
    pbz, zpz = pbs['ps1z']
    pbk, zpk = pbs['Kep']

    mg = []
    mr = []
    mi = []
    mz = []
    mk = []
    # construct mags
    ind = []
    red = {}
    for modelname in Models:
        model = Models[modelname]
        model = S.ArraySpectrum(model.wave, 
                                apply(fitzpatrick99(model.wave.astype('double'),a_v,3.1),model.flux), 
                                name=modelname)
        mg += [source_synphot.passband.synphot(model, pbg,zpg)]
        mr += [source_synphot.passband.synphot(model, pbr,zpr)]
        mi += [source_synphot.passband.synphot(model, pbi,zpi)]
        mz += [source_synphot.passband.synphot(model, pbz,zpz)]
        mk += [source_synphot.passband.synphot(model, pbk,zpk)]

    mg = np.array(mg)
    mr = np.array(mr)
    mi = np.array(mi)
    mz = np.array(mz)
    mk = np.array(mk)
    good = np.isfinite(mg) & np.isfinite(mr) & np.isfinite(mi) & np.isfinite(mz) & np.isfinite(mk)
    d = {'g':mg[good], 'r':mr[good],'i':mi[good],'z':mz[good],'k':mk[good]}
    return d




def Plotter(K, Colours, Compare, Channel, Camp, fitfilt, Save, Residuals = False, Close = True):

    plt.figure(figsize=(10,6))
    plt.suptitle(fitfilt + ' band \n Channel ' + str(Channel) + ', $Zp=$ ' + str(np.round(K,3)[0]))
    for i in range(len(Compare)):
        X,Y = Compare[i]
        ob_x, ob_y, locus = Get_lcs(X,Y,K,Colours,fitfilt)

        ind = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))[0]
        ob_x = ob_x[:,ind]
        ob_y = ob_y[:,ind]
        
        
        plt.subplot(2,2,i+1)
        plt.xlabel(X)
        plt.ylabel(Y)
        if i == 2:
            if Residuals:
                dist = Dist_tensor(X,Y,K,Colours,fitfilt,Tensor = True)
                
                plt.errorbar(ob_x[0,:],dist,ob_y[1,:],fmt='.',alpha=0.4,label='Observed')
                plt.axhline(0,ls='--',color='k',label='Calspec model')
                plt.legend(loc='upper center', bbox_to_anchor=(1.5, .5))
                plt.ylabel(r'('+ Y +')obs - (' + Y + ')syn')
                #plt.ylim(-.6, .6)
            else:
                plt.errorbar(ob_x[0,:],ob_y[0,:],ob_y[1,:],fmt='.',alpha=0.4,label='Observed')
                ind = (locus[0,:] > 0.2) & (locus[0,:] < 0.8)
                plt.plot(locus[0,ind],locus[1,ind],label='Calspec model')
                plt.legend(loc='upper center', bbox_to_anchor=(1.5, .5))
                plt.xlim(0.2, .8)
                #plt.ylim(-.5, 1)
        else:
            if Residuals:
                dist = Dist_tensor(X,Y,K,Colours,fitfilt,Tensor = True)
                plt.axhline(0,ls='--',color='k',label='offset = {}'.format(np.round(K[0],4)))
                plt.errorbar(ob_x[0,:],dist,ob_y[1,:],fmt='.',alpha=0.4)
                #plt.legend()
                plt.ylabel(r'('+ Y +')obs - (' + Y + ')syn')
                #plt.hist(dist.flatten(),bins=100)
                #plt.ylim(-.6, .6)
            else:
                plt.errorbar(ob_x[0,:],ob_y[0,:],ob_y[1,:],fmt='.',alpha=0.4)
                ind = (locus[0,:] > 0.2) & (locus[0,:] < 0.8)
                plt.plot(locus[0,ind],locus[1,ind])
                #plt.xlim(-0.5, 1)
                plt.xlim(0.2, .8)
                #plt.ylim(-.5, 1)
    plt.subplots_adjust(wspace=.25,hspace=.25)
    
    path = Save + 'C'+str(Camp) +'/' + fitfilt+ '/' 
    Save_space(path)
    if Residuals:
        plt.savefig(path+ fitfilt + '_fit_ch' + str(Channel) +'_residual.png')
    else:
        plt.savefig(path+ fitfilt + '_fit_ch' + str(Channel) + '.png')
    if Close:
        plt.close()
    
def Dot_prod_error(x,y,Model):
    """
    Calculate the error projection in the direction of a selected point.
    """
    #print(Model.shape)
    adj = y[0,:] - Model[1,:]
    op = x[0,:] - Model[0,:]
    #print(adj.shape,op.shape)
    hyp = np.sqrt(adj**2 + op**2)
    costheta = adj / hyp
    yerr_proj = abs(y[1,:] * costheta)
    xerr_proj = abs(x[1,:] * costheta)
    
    proj_err = yerr_proj + xerr_proj
    #print(proj_err)
    return proj_err 

def Calculate_distance(data,trend):
    x = np.zeros((data.shape[1],trend.shape[1])) + data[0,:,np.newaxis]
    x -= trend[0,np.newaxis,:]
    y = np.zeros((data.shape[1],trend.shape[1])) + data[1,:,np.newaxis]
    y -= trend[1,np.newaxis,:]

    dist = np.sqrt(x**2 + y**2)

    minind = np.nanargmin(abs(dist),axis=1)
    #proj_err = Dot_prod_error(ob_x,ob_y,locus[:,minind])
    mindist = np.nanmin(abs(dist),axis=1)
    sign = (data[1,:] - trend[1,minind])
    sign = sign / abs(sign)

    resid = mindist * sign
    return resid

def Dist_tensor(X,Y,K,Colours,fitfilt='',Tensor=False,Plot = False):
    ob_x, ob_y, locus = Get_lcs(X,Y,K,Colours,fitfilt)
    
    ind = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))[0]
    indo = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))
    ob_x = ob_x[:,ind]
    ob_y = ob_y[:,ind]
    
    
    if Plot:
        plt.figure()
        plt.title(X + ' ' + Y)
        plt.plot(ob_x[0,:],ob_y[0,:],'.')
        plt.plot(locus[0,:],locus[1,:])
    #print(ob_x.shape)
    #print('x ',ob_x.shape[1])

    x = np.zeros((ob_x.shape[1],locus.shape[1])) + ob_x[0,:,np.newaxis]
    x -= locus[0,np.newaxis,:]
    y = np.zeros((ob_y.shape[1],locus.shape[1])) + ob_y[0,:,np.newaxis]
    y -= locus[1,np.newaxis,:]

    dist_tensor = np.sqrt(x**2 + y**2)
    #print(np.nanmin(dist_tensor,axis=1))
    #print(X + Y +' dist ',dist_tensor.shape)
    if len(dist_tensor) > 0:
        minind = np.nanargmin(abs(dist_tensor),axis=1)
        mindist = np.nanmin(abs(dist_tensor),axis=1)
        sign = (ob_y[0,:] - locus[1,minind])
        sign = sign / abs(sign)

        eh = mindist * sign
    
        proj_err = Dot_prod_error(ob_x,ob_y,locus[:,minind])
        #print('mindist ',mindist)
        if Tensor:
            return eh
        if len(mindist) > 0:
            #print('thingo',np.nanstd(mindist))
            residual = np.nansum(abs(mindist)) #/ proj_err)
        else:
            #print('infs')
            residual = np.inf
    else:
        if Tensor:
            return []
        residual = np.inf
        #residual += 100*np.sum(np.isnan(dist))
    #print(residual)
    cut_points = len(indo) - len(ind)
    return residual + cut_points * 100



import copy
def Get_lcs(X,Y,K,Colours,fitfilt = ''):
    keys = np.array(list(Colours.keys()))

    xind = 'mod ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'mod ' + Y == keys
    y = Colours[keys[yind][0]]

    #x_interp = np.arange(np.nanmin(x),0.8,0.01)
    #inter = interpolate.interp1d(x,y)
    #l_interp = inter(x_interp)
    locus = np.array([x,y])

    xind = 'obs ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'obs ' + Y == keys
    y = Colours[keys[yind][0]]
    c1,c2 = X.split('-')
    c3,c4 = Y.split('-')
    # parameters
    ob_x = x.copy() 
    ob_y = y.copy() 

    if c1 == fitfilt: ob_x[0,:] += K
    if c2 == fitfilt: ob_x[0,:] -= K

    if c3 == fitfilt: ob_y[0,:] += K
    if c4 == fitfilt: ob_y[0,:] -= K
    return ob_x, ob_y, locus

def sigma_mask(data,error= None,sigma=3,Verbose= False):
    if type(error) == type(None):
        error = np.zeros(len(data))
    
    calcaverage = sigmacut.calcaverageclass()
    calcaverage.calcaverage_sigmacutloop(data,Nsigma=sigma
                                         ,median_firstiteration=True,saveused=True)
    if Verbose:
        print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
    return calcaverage.clipped


def Cut_data(K,Data,Model,Compare,Extinction,Band = '',Plot=False):
    Colours = Make_colours(Data,Model,Compare,Extinction, Redden = False)
    keys = list(Colours.keys())
    bads = []
    for X,Y in Compare:
        #X = 'g-r'
        #Y = 'i-z'
        dist = Dist_tensor(X,Y,K,Colours,Band,True)
        if len(dist) > 5:

            ob_x, ob_y, locus = Get_lcs(X,Y,K,Colours,Band)
            ob_x2, ob_y2, locus = Get_lcs(X,Y,K,Colours,Band)
            ind = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))[0]

            #if X == 'g-r':
            #   ind = np.where((ob_x[0,:] <= .9) & (ob_x[0,:] >= 0.2) & (ob_y[1,:] < 0.5))[0]
            #elif X == 'r-i':
            #   ind = np.where((ob_x[0,:] <= .6) & (ob_x[0,:] >= 0) & (ob_y[1,:] < 0.5))[0]
            ob_x = ob_x[:,ind]
            ob_y = ob_y[:,ind]

            #print(Colours)
            #bad = []
            #print(type(dist))
            #print('std',np.nanstd(dist.flatten()))
            dist = dist.flatten()
            finiteinds = np.where(np.isfinite(dist))[0]
            #dist[~np.isfinite(dist)] = np.nan
            #print(ob_y)
            bad = sigma_mask(dist,sigma=2)
            bad = finiteinds[bad]
            #for i in range(len(dist)):
                #print(dist[i],ob_y[1,i])
             #   if abs(dist[i]) > 1:
                    #print(dist,ob_y[0,i])
              #   bad += [i]
              #  if abs(dist[i]) > 10*ob_y[1,i]:

               #     bad += [i]
            if Plot:
                plt.figure()
                plt.errorbar(ob_x2[0,:],ob_y2[0,:],yerr = ob_y2[1,:],fmt='.')
                plt.errorbar(locus[0,:],locus[1,:])
                #plt.errorbar(ob_x[0,bad],ob_y[0,bad],yerr = ob_y[1,bad],fmt='.')
                plt.errorbar(ob_x2[0,ind[bad]],ob_y2[0,ind[bad]],yerr = ob_y2[1,ind[bad]],fmt='r.')
                #plt.xlim(.5,0.9)
            bads += [ind[bad]]
    bads = np.array(bads)
    return bads

def Parms_dict(K):
    num = len(K)
    
    k =np.zeros(6)
    if num ==1:
        k[0] = K[0]
        k[1:] = 0
        
    elif num == 6:
        k = K
    Params = {}
    Params['fitzp'] = k[0]
    Params['Ak'] = k[1]
    Params['Ag'] = k[2]
    Params['Ar'] = k[3]
    Params['Ai'] = k[4]
    Params['Az'] = k[5]
    
    return Params




def Make_colours(Data, Model, Compare, Extinction = 0, Redden = False,Tonry=False):
    R = {'g': 3.518, 'r':2.617, 'i':1.971, 'z':1.549, 'y': 1.286, 'k':2.431,'tess':1.809}#'z':1.549} # value from bayestar
    colours = {}
    for x,y in Compare:
        colours['obs ' + x] = np.array([Data[x.split('-')[0]+'MeanPSFMag'].values - Data[x.split('-')[1]+'MeanPSFMag'].values,
                                        Data[x.split('-')[0]+'MeanPSFMagErr'].values - Data[x.split('-')[1]+'MeanPSFMagErr'].values])
        colours['obs ' + y] = np.array([Data[y.split('-')[0]+'MeanPSFMag'].values - Data[y.split('-')[1]+'MeanPSFMag'].values,
                                        Data[y.split('-')[0]+'MeanPSFMagErr'].values - Data[y.split('-')[1]+'MeanPSFMagErr'].values])
        if Tonry:
            colours['mod ' + x] = Model[:,0]
            colours['mod ' + y] = Model[:,1]
            # colour cut to remove weird top branch present in C2
            #if (y == 'g-r'):
            #    ind = colours['obs g-r'][0,:] > 1.4
            #    colours['obs g-r'][:,ind] = np.nan
            #    colours['obs r-i'][:,ind] = np.nan
        else:

            xx = Model[x.split('-')[0]] - Model[x.split('-')[1]]
            yy = Model[y.split('-')[0]] - Model[y.split('-')[1]]
            ind = xx.argsort()
            xx = xx[ind]
            yy = yy[ind]
            spl = UnivariateSpline(xx, yy)
            c_range = np.arange(xx[0],0.8,0.01)
            colours['mod ' + x] = c_range
            colours['mod ' + y] = spl(c_range)
        
        if Redden:
            colours['mod ' + x] += Extinction*((R[x.split('-')[0]] - R[x.split('-')[1]]))
            colours['mod ' + y] += Extinction*(R[y.split('-')[0]] - R[y.split('-')[1]])
        else:
            colours['obs ' + x] -= Extinction*((R[x.split('-')[0]] - R[x.split('-')[1]]))
            colours['obs ' + y] -= Extinction*(R[y.split('-')[0]] - R[y.split('-')[1]])
    return colours 

def SLR_residual_multi(K,Data,Model,Compare,Ex,Band=''):
    #print('guess ', K)
    #params = Parms_dict(K)
    Colours = Make_colours(Data, Model, Compare, Extinction = Ex, Redden = False)
    
    res = 0
    #print(K)
    for x,y in Compare:
        residual = Dist_tensor(x,y,K,Colours,Band,Tensor=True,Plot=False)
        if len(residual)>0:
            res += np.nansum(abs(residual))
        else:
            res = np.inf
    #print('residual ', res)
    return res
    
def Fit_zeropoint(Data,Model,Compare,Ex,Band):
    k = 0
    res = minimize(SLR_residual_multi,k,args=(Data,Model,Compare,Ex,Band))
    #Colours = Make_colours(Data,Model,Compare,Extinction = Ex)
    bads = Cut_data(k,Data,Model,Compare,Ex,Band=Band)

    inds = np.ones(len(Data))
    for b in bads:
        inds[b] = 0
    data = Data.iloc[inds > 0]
    #print(inds)
    res = minimize(SLR_residual_multi,k,args=(data,Model,Compare,Ex,Band))
    return res.x, data


