def Tonry_clip(Colours):
    """
    Use the Tonry 2012 PS1 splines to sigma clip the observed data.
    """
    tonry = np.loadtxt('Tonry_splines.txt')
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
    mins = np.nanmin(dd,axis=1)
    # Sigma clip the distance data
    sig = sigma_mask(mins)
    # return sigma clipped mask
    return ~sig

def Tonry_residual(Colours):
    """
    Calculate the residuals of the observed data from the Tonry et al 2012 PS1 splines.
    """
    tonry = np.loadtxt('Tonry_splines.txt')
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
    Colours = Make_colours(Data,Model,Compare,Extinction = K)
    res = Tonry_residual(Colours)
    return res

def Tonry_reduce(Data,Camp):
    '''
    Uses the Tonry et al. 2012 PS1 splines to fit dust and find all outliers.
    '''
    data = copy.deepcopy(Data)
    tonry = np.loadtxt('Tonry_splines.txt')
    compare = np.array([['r-i','g-r'],['r-i','i-z']])      
    cind =  ((data['campaign'].values == Camp))
    dat = data.iloc[cind]
    clips = []
    for i in range(2):
        if i == 0:
            k0 = 0
        else:
            k0 = res.x
        res = minimize(Tonry_fit,k0,args=(dat,tonry,compare),method='Nelder-Mead')
        
        colours = Make_colours(dat,tonry,compare,Extinction = res.x)
        clip = Tonry_clip(colours)
        clips += [clip]
        dat = dat.iloc[clip]
        print('Pass ' + str(i+1) + ': '  + str(np.round(res.x[0],4)))
    clips[0][clips[0]] = clips[1]
    return res.x, clips[0]