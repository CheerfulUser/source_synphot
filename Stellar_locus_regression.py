import source_synphot.passband
import source_synphot.io
import source_synphot.source
import astropy.table as at
from collections import OrderedDict
import pysynphot as S
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
%matplotlib notebook

import sys
sys.path.append('../Sigma_clip/')
import sigmacut

#from zp_fit import *
from scipy.optimize import minimize
from astropy.stats import sigma_clip
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
def myround(x, prec=2, base=.5):
    return round(base * round(float(x)/base),prec)


def Plotter(K,Colours,Compare,Channel,fitfilt, Residuals = False, Close = True):
    Params = Parms_dict(K)
    plt.figure()#figsize=(10,6))
    #plt.suptitle('Channel ' + str(Channel) + ', Zp = ' + str(np.round(K[0],3)) + 
     #            '\n' + 'Ak ={}, Ag = {}, Ar = {}, Ai ={}'.format(np.round(Params['Ak'],3),
     #                                                             np.round(Params['Ag'],3),
     #                                                             np.round(Params['Ar'],3),
     #                                                             np.round(Params['Ai'],3)))
    for i in range(len(Compare)):
        X,Y = Compare[i]
        keys = np.array(list(Colours.keys()))
        xind = 'mod ' + X == keys
        x = Colours[keys[xind][0]]
        yind = 'mod ' + Y == keys
        y = Colours[keys[yind][0]]

        x_interp = np.arange(np.nanmin(x),0.9,0.01)
        inter = interpolate.interp1d(x,y)
        l_interp = inter(x_interp)
        locus = np.array([x_interp,l_interp])
        ind  = np.where((locus[0,:] <= .9) & (locus[0,:] >= 0.2))[0]
        locus = locus[:,ind]
        # observed
        xind = 'obs ' + X == keys
        x = Colours[keys[xind][0]]
        yind = 'obs ' + Y == keys
        y = Colours[keys[yind][0]]
        #print(X,Y)
        c1,c2 = X.split('-')
        c3,c4 = Y.split('-')
        #print(c1,c2,c3,c4)

        # parameters
        ob_x = x.copy() 
        ob_x[0,:] += Params['A'+c1] - Params['A'+c2]
        if c1 == fitfilt: ob_x += Params['fitzp']
        if c2 == fitfilt: ob_x -= Params['fitzp']

        ob_y = y.copy() 
        ob_y[0,:] += Params['A'+c3] - Params['A'+c4]
        if c3 == fitfilt: ob_y[0,:] += Params['fitzp']
        if c4 == fitfilt: ob_y[0,:] -= Params['fitzp']

        ind = np.where((Colours['obs g-r'][0,:] <= .9) & (Colours['obs g-r'][0,:] >= 0.2))[0]
        ob_x = ob_x[:,ind]
        ob_y = ob_y[:,ind]
        
        
        #plt.subplot(2,2,i+1)
        plt.xlabel(X)
        plt.ylabel(Y)
        if i == 2:
            if Residuals:
                dist = Dist_tensor(X,Y,Params,Colours,fitfilt,Tensor = True)
                
                plt.errorbar(ob_x[0,:],dist,ob_y[1,:],fmt='.',alpha=0.4,label='Observed')
                plt.axhline(0,ls='--',color='k',label='Model')
                plt.legend(loc='upper center', bbox_to_anchor=(1.5, .5))
                plt.ylim(-.6, .6)
            else:
                plt.errorbar(ob_x[0,:],ob_y[0,:],ob_y[1,:],fmt='.',alpha=0.4,label='Observed')
                plt.plot(locus[0,:],locus[1,:],label='Model')
                plt.legend(loc='upper center', bbox_to_anchor=(1.5, .5))
                plt.xlim(0.2, .9)
                #plt.ylim(-.5, 1)
        else:
            if Residuals:
                dist = Dist_tensor(X,Y,Params,Colours,fitfilt,Tensor = True)
                plt.axhline(0,ls='--',color='k',label='offset = {}'.format(np.round(K[0],4)))
                plt.errorbar(ob_x[0,:],dist,ob_y[1,:],fmt='.',alpha=0.4)
                plt.legend()
                plt.ylabel(r'('+ Y +')obs - (' + Y + ')syn')
                #plt.hist(dist.flatten(),bins=100)
                #plt.ylim(-.6, .6)
            else:
                plt.errorbar(ob_x[0,:],ob_y[0,:],ob_y[1,:],fmt='.',alpha=0.4)
                plt.plot(locus[0,:],locus[1,:])
                #plt.xlim(-0.5, 1)
                plt.xlim(0.2, .9)
                #plt.ylim(-.5, 1)
    plt.subplots_adjust(wspace=.25,hspace=.2)
    if Residuals:
        plt.savefig('./figs/fit_i/'+ fitfilt + '_fit_' + Y + '_residual'+ '_camp' + str(Channel) +'.png')
    else:
        plt.savefig('./figs/fit_i/'+ fitfilt + Y + '_camp' + str(Channel) + '.png')
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

def Dist_tensor(X,Y,Params,Colours,fitfilt,Tensor=False,Plot = False):
    keys = np.array(list(Colours.keys()))
    xind = 'mod ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'mod ' + Y == keys
    y = Colours[keys[yind][0]]
    
    x_interp = np.arange(np.nanmin(x),0.9,0.01)
    inter = interpolate.interp1d(x,y)
    l_interp = inter(x_interp)
    locus = np.array([x_interp,l_interp])
    # observed
    xind = 'obs ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'obs ' + Y == keys
    y = Colours[keys[yind][0]]
    #print(X,Y)
    c1,c2 = X.split('-')
    c3,c4 = Y.split('-')
    #print(c1,c2,c3,c4)
    

    # parameters
    ob_x = x.copy() 
    ob_y = y.copy() 
    indo = np.where((Colours['obs g-r'][0,:] <= .9) & (Colours['obs g-r'][0,:] >= 0.2))
    
    ob_x[0,:] += Params['A'+c1] - Params['A'+c2]
    if c1 == fitfilt: ob_x[0,:] += Params['fitzp']
    if c2 == fitfilt: ob_x[0,:] -= Params['fitzp']

    
    ob_y[0,:] += Params['A'+c3] - Params['A'+c4]
    if c3 == fitfilt: ob_y[0,:] += Params['fitzp']
    if c4 == fitfilt: ob_y[0,:] -= Params['fitzp']
    
    ind = np.where((Colours['obs g-r'][0,:] <= .9) & (Colours['obs g-r'][0,:] >= 0.2))[0]
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
def Get_lcs(X,Y,Params,Colours,fitfilt):
    keys = np.array(list(Colours.keys()))

    xind = 'mod ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'mod ' + Y == keys
    y = Colours[keys[yind][0]]

    x_interp = np.arange(np.nanmin(x),0.9,0.01)
    inter = interpolate.interp1d(x,y)
    l_interp = inter(x_interp)
    locus = np.array([x_interp,l_interp])

    xind = 'obs ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'obs ' + Y == keys
    y = Colours[keys[yind][0]]
    c1,c2 = X.split('-')
    c3,c4 = Y.split('-')
    # parameters
    ob_x = x.copy() 
    ob_y = y.copy() 

    ob_x[0,:] += Params['A'+c1] - Params['A'+c2]
    if c1 == fitfilt: ob_x[0,:] += Params['fitzp']
    if c2 == fitfilt: ob_x[0,:] -= Params['fitzp']


    ob_y[0,:] += Params['A'+c3] - Params['A'+c4]
    if c3 == fitfilt: ob_y[0,:] += Params['fitzp']
    if c4 == fitfilt: ob_y[0,:] -= Params['fitzp']
    return ob_x, ob_y, locus

def sigma_mask(data,error= None,sigma=3,Verbose= False):
    if type(error) == type(None):
        error = np.zeros(len(data))
    
    calcaverage = sigmacut.calcaverageclass()
    calcaverage.calcaverage_sigmacutloop(data,verbose=2,Nsigma=sigma
                                         ,median_firstiteration=True,saveused=True)
    #if Verbose:
    print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
    return calcaverage.clipped


def Cut_data(K,Colours,Compare,fitfilt,Plot=False):
    Params = Parms_dict(K)
    c_cut = copy.deepcopy(Colours)
    for X,Y in Compare:
        #X = 'g-r'
        #Y = 'i-z'
        dist = Dist_tensor(X,Y,Params,Colours,fitfilt,True)
        if len(dist) > 0:

            ob_x, ob_y, locus = Get_lcs(X,Y,Params,Colours,fitfilt)
            ob_x2, ob_y2, locus = Get_lcs(X,Y,Params,Colours,fitfilt)
            ind = np.where((Colours['obs g-r'][0,:] <= .9) & (Colours['obs g-r'][0,:] >= 0.2))[0]

            #if X == 'g-r':
            #    ind = np.where((ob_x[0,:] <= .9) & (ob_x[0,:] >= 0.2) & (ob_y[1,:] < 0.5))[0]
            #elif X == 'r-i':
            #    ind = np.where((ob_x[0,:] <= .6) & (ob_x[0,:] >= 0) & (ob_y[1,:] < 0.5))[0]
            ob_x = ob_x[:,ind]
            ob_y = ob_y[:,ind]


            #bad = []
            #print(type(dist))
            #print('std',np.nanstd(dist.flatten()))
            dist = dist.flatten()
            finiteinds = np.where(np.isfinite(dist))[0]
            #dist[~np.isfinite(dist)] = np.nan
            bad = sigma_mask(dist,error=ob_y[:,1],sigma=3)
            bad = finiteinds[bad]
            #for i in range(len(dist)):
                #print(dist[i],ob_y[1,i])
             #   if abs(dist[i]) > 1:
                    #print(dist,ob_y[0,i])
              #      bad += [i]
              #  if abs(dist[i]) > 10*ob_y[1,i]:

               #     bad += [i]
            if Plot:
                plt.figure()
                plt.errorbar(ob_x2[0,:],ob_y2[0,:],yerr = ob_y2[1,:],fmt='.')
                plt.errorbar(locus[0,:],locus[1,:])
                #plt.errorbar(ob_x[0,bad],ob_y[0,bad],yerr = ob_y[1,bad],fmt='.')
                plt.errorbar(ob_x2[0,ind[bad]],ob_y2[0,ind[bad]],yerr = ob_y2[1,ind[bad]],fmt='r.')
                #plt.xlim(.5,0.9)

            for key in keys:
                if 'obs' in key:
                    #print(dist[bad],6*c_cut[key][1, main_ind[bad]])
                    c_cut[key][:, ind[bad]] = np.nan
                    #print('killed ', key)
            #print(colours[keys[xind][0]][:, main_ind[bad]])
    return c_cut

def Parms_dict(K):
    num = len(K)
    
    k =np.zeros(6)
    if num ==1:
        k[0] = K[0]
        k[1:] = 0
    else:
        k = K
    Params = {}
    Params['fitzp'] = k[0]
    Params['Ak'] = k[1]
    Params['Ag'] = k[2]
    Params['Ar'] = k[3]
    Params['Ai'] = k[4]
    Params['Az'] = k[4]
    
    return Params

def SLR_fit_multi(K,Colours,Compare,fitfilt,Second=False):
    
    params = Parms_dict(K)
    res = 0
    #print(K)
    for x,y in Compare:
        residual = Dist_tensor(x,y,params,Colours,fitfilt,Tensor=True)
        if len(residual)>0:
            res += np.nansum(abs(residual))
        else:
            res = np.inf
    #print('residual ', res)
    return res
    