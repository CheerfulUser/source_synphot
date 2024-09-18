import numpy as np

def mag2flux(mag,zp=25):
    flux = 10**(-1/2.5*(mag-zp))
    return flux


def u_comp(g,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
    cg = 0.58906083; cp = 1.03347526
    fit = (cg*g)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt


def g_comp(g,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
    cg = 0.99169699; cp = 0.02816213
    fit = (cg*g)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt

def r_comp(r,g,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        r = mag2flux(r,zp)
        i = mag2flux(i,zp)
    cr = 0.99468739; cp = -0.07174337
    fit = (cr*r)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt


def i_comp(i,g,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
    ci = 0.98325917; cp = -0.06073959
    fit = (ci*i)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt

def z_comp(z,y,g,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
        z = mag2flux(z,zp)
        y = mag2flux(y,zp)
    cz = 0.42938712; cy = 0.57355253; cp = 0.00767245
    fit = (cz*z + cy*y)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt