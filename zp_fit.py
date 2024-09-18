import source_synphot.passband
import source_synphot.io
import source_synphot.source
import astropy.table as at
from collections import OrderedDict
import pysynphot as S
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from scipy.optimize import minimize
from astropy.stats import sigma_clip
import pandas as pd


def SLR_fit(K, Observed, Locus,Second):
    
    if len(K) > 1:
        zp = K[0]
        A = K[1:]
    else:
        zp = K[0]
        A = np.zeros(3)
    x_interp = np.arange(np.nanmin(Locus[0,:]),1,0.01)
    inter = interpolate.interp1d(Locus[0,:],Locus[1,:])
    l_interp = inter(x_interp)
    locus = np.array([x_interp,l_interp])
    
    c = Observed.copy() 
    ind = np.where((c[0,:] <= 1) & (c[0,:] > 0.5))[0]
    c = c[:,ind]
    c[1,:] = c[1,:] + zp + A[2] - A[1]
    c[0,:] = c[0,:] + A[0] - A[1]
    
    #print(c[1,:])
    dist_tensor = []
    for i in range(c.shape[1]):
        dist = np.sqrt((c[0,i] - locus[0,:])**2 + (c[1,i] - locus[1,:])**2) 
        if Second:
            #print('cut')
            # Quality control
            #dist[dist > 6*(c[2,i])] = np.nan
            dist[dist > 1] = np.nan
        #else:
            #print('first')
        # add the normalised distance
        dist_tensor += [dist / c[2,i]]
    dist_tensor = np.array(dist_tensor)
    if len(dist_tensor) > 2:
        residual = np.nansum(np.nanmin(abs(dist_tensor),axis=1))
    else:
        residual = 1000
    #print(residual)
    return residual
    

def Dist_tensor(X,Y,Params,Colours,Second,Plot = False):
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
    ob_x[0,:] += Params['A'+c1] - Params['A'+c2]
    if c1 == 'k': ob_x += Params['kzp']
    if c2 == 'k': ob_x -= Params['kzp']

    ob_y = y.copy() 
    ob_y[0,:] += Params['A'+c3] - Params['A'+c4]
    if c3 == 'k': ob_y[0,:] += Params['kzp']
    if c4 == 'k': ob_y[0,:] -= Params['kzp']
    
    ind = np.where((ob_x[0,:] <= 1) & (ob_x[0,:] >= 0.5))[0]
    ob_x = ob_x[:,ind]
    ob_y = ob_y[:,ind]
    if Plot:
        plt.figure()
        plt.title(X + ' ' + Y)
        plt.plot(ob_x[0,:],ob_y[0,:],'.')
        plt.plot(locus[0,:],locus[1,:])
    #print(ob_x.shape)
    dist_tensor = []
    for i in range(ob_y.shape[1]):
        dist = np.sqrt((ob_x[0,i] - locus[0,:])**2 + (ob_y[0,i] - locus[1,:])**2) 
        if Second:
            # Quality control
            #dist[dist > 6*(c[2,i])] = np.nan
            dist[dist > 1] = np.nan
        # add the normalised distance
        dist_tensor += [dist / ob_y[1,i]]
        if (c1 == 'k') | (c2 == 'k') | (c3 == 'k') | (c4 == 'k'):
            dist_tensor = dist_tensor * 2
    dist_tensor = np.array(dist_tensor)
    #print(dist_tensor.shape)
    if len(dist_tensor) > 20:
        residual = np.nansum(np.nanmin(abs(dist_tensor),axis=1))
    else:
        residual = 1000000
    #print(residual)
    return residual
    
def SLR_fit_multi(K,Colours,Compare,Second):
    
    params = {}
    params['kzp'] = K[0]
    params['Ak'] = K[1]
    params['Ag'] = K[2]
    params['Ar'] = K[3]
    params['Ai'] = K[4]
    res = 0
    for x,y in Compare:
        res += Dist_tensor(x,y,params,Colours,Second)
    return res


def Plotter(K,Colours,Compare,Channel):
    Params = {}
    Params['kzp'] = K[0]
    Params['Ak'] = K[1]
    Params['Ag'] = K[2]
    Params['Ar'] = K[3]
    Params['Ai'] = K[4]
    plt.figure(figsize=(10,6))
    plt.suptitle('Channel ' + str(Channel) + ', Zp = ' + str(np.round(K[0],2)))
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
        if c1 == 'k': ob_x += Params['kzp']
        if c2 == 'k': ob_x -= Params['kzp']

        ob_y = y.copy() 
        ob_y[0,:] += Params['A'+c3] - Params['A'+c4]
        if c3 == 'k': ob_y[0,:] += Params['kzp']
        if c4 == 'k': ob_y[0,:] -= Params['kzp']

        ind = np.where((ob_x[0,:] <= 1) & (ob_x[0,:] > -0.5) & ((ob_y[1,:] < 5)))[0]
        ob_x = ob_x[:,ind]
        ob_y = ob_y[:,ind]
        
        plt.subplot(2,2,i+1)
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.errorbar(ob_x[0,:],ob_y[0,:],ob_y[1,:],fmt='.',alpha=0.4)
        plt.plot(locus[0,:],locus[1,:])
        plt.xlim(-0.5, 1)
        plt.ylim(-.5, 1)
    plt.subplots_adjust(wspace=.2,hspace=.2)
    plt.savefig('./figs/Fit_channel_' + str(j) + '.png')
    plt.close()


'''
compare = np.array([['g-r','k-r'],['g-r','k-i'],['g-r','r-i'],['r-i','g-k']])
zps = np.zeros((84,100)) * np.nan
for j in range(84):
    j += 1
    cind =  ((data['campaign'].values == 16) & (data['Channel'].values == j) & (kbe < 100))

    colours = {}
    colours['obs g-r'] = np.array([(gb - rb)[cind], (gbe + rbe)[cind]])
    colours['obs k-i'] = np.array([(kb - ib)[cind], (kbe + ibe)[cind]])
    colours['obs k-r'] = np.array([(kb - rb)[cind], (kbe + rbe)[cind]])
    colours['obs r-i'] = np.array([(rb - ib)[cind], (rbe + ibe)[cind]])
    colours['obs g-k'] = np.array([(gb - kb)[cind], (gbe + kbe)[cind]])
    colours['mod g-r'] = col1[good]
    colours['mod r-i'] = col2[good]
    colours['mod k-r'] = col3[good]
    colours['mod k-i'] = col4[good]
    colours['mod g-k'] = col5[good]
    #print(obs.shape)
    zp0 = 25.4
    Ak0 = Ag0 = Ar0 = Ai0 = 0
    k0 = np.array([zp0,Ak0,Ag0,Ar0,Ai0])
    bds = [(22,27),(-2,2),(-2,2),(-2,2),(-2,2)]
    res = minimize(SLR_fit_multi,k0,args=(colours,compare,False),bounds=bds)
    K0 = res.x
    #res = minimize(SLR_fit_multi,k0,args=(colours,compare,True),bounds=bds)
    if len(res.x) < 2:
        zps[j] = res.x
    else:
        zps[j-1] = res.x[0]
        print(res.x[0])

    Plotter(res.x,colours,compare,j)

'''