import numpy as np

def mag2flux(mag,zp=25):
    flux = 10**(-1/2.5*(mag-zp))
    return flux

coeffs = {'u': [0.8,0,0,0,0,0.1],
          'g': [0.98752191, 0, 0 , 0, 0, 0.04839131],
          'r': [0, 0.99921576, 0, 0, 0. -0.0135116],
          'i': [0, 0, 0.99086808,0 ,0, -0.0318901],
          'z': [0, 0, 0, 0.5, 0.5,0]
          'B': [0.85309989, 0, 0, 0, 0, 0.40963701],
          'V': [0.03637554, 0.95882605, 0, 0, 0, 0.29828244]
          'R': [0, 0.7015572, 0.30016923, 0, 0, 0.00286492]
          'I': [0, 0, 0.97131081, 0.49998139, 0, -0.10969656]
          'Z': [0, 0, 0, 0.5, 0.5,0]}



def Kepcam_comp(g,r,i,z,y,band,ebv=0):
    gr = g - r
    eg, e = R_val('g',gr=gr,ext=ebv,instrument='ps1'); er, e = R_val('r',gr=gr,ext=ebv,instrument='ps1')
    ei, e = R_val('i',gr=gr,ext=ebv,instrument='ps1'); ez, e = R_val('z',gr=gr,ext=ebv,instrument='ps1')
    ey, e = R_val('y',gr=gr,ext=ebv,instrument='ps1'); et, e = R_val('tess',gr=gr,ext=ebv,instrument='ps1')
    eg = eg  * ebv; er = er  * ebv; ei = ei  * ebv; ez = ez  * ebv
    ey = ey  * ebv; et = et  * ebv

    zp = 25
    g = mag2flux(g - eg,zp)
    r = mag2flux(r - er,zp)
    i = mag2flux(i - ei,zp)
    z = mag2flux(z - ez,zp)
    y = mag2flux(y - ey,zp)

    coeff = coeffs[band]

    comp = (coeff[0]*g + coeff[1]*r + coeff[2]*i + coeff[3]*z + coeff[4]*y)*(g/i)**coeff[5]

    comp = -2.5*np.log10(fit) + zp
    eb, e = R_val(band,gr=gr,ext=ebv,instrument='kepcam')
    comp += eb
    return comp 


def Kepcam_u(g,i,mags=True):
    print('!!!WARNING!!! this relation is VERY inacurate!')
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
    cg = 0.8; cp = 0.1
    fit = (cg*g)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt


def Kepcam_g(g,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
    cg = 0.98752191; cp = 0.04839131
    fit = (cg*g)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt

def Kepcam_r(r,g,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        r = mag2flux(r,zp)
        i = mag2flux(i,zp)
    cr = 0.99921576; cp = -0.0135116
    fit = (cr*r)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt


def Kepcam_i(i,g,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
    ci = 0.99086808; cp = -0.0318901
    fit = (ci*i)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt

def Kepcam_z(z,y,g,i,mags=True):
    print('!!!WARNING!!! this relation is very inacurate (std = 0.1 mag)!')
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
        z = mag2flux(z,zp)
        y = mag2flux(y,zp)
    cz = 0.5; cy = 0.5; cp = 0
    fit = (cz*z + cy*y)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt



def Kepcam_B(g,i,mags=True):
    print('!!!WARNING!!! this relation is inacurate (std = 0.05 mag)!')
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
    cg = 0.85309989; cp = 0.40963701
    fit = (cg*g)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt


def Kepcam_V(g,r,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        r = mag2flux(r,zp)
        i = mag2flux(i,zp)
    cg = 0.03637554; cr = 0.95882605; cp = 0.29828244
    fit = (cg*g + cr*r)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt

def Kepcam_R(r,g,i,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        r = mag2flux(r,zp)
        i = mag2flux(i,zp)
    cr = 0.7015572; ci = 0.30016923; cp = 0.00286492
    fit = (cr*r + ci*i)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt


def Kepcam_I(i,z,g,mags=True):
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
        z = mag2flux(z,zp)
    ci = 0.97131081; cz = 0.49998139; cp = -0.10969656
    fit = (ci*i + cz*z)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt

def Kepcam_Z(z,y,g,i,mags=True):
    print('!!!WARNING!!! this relation is very inacurate (std = 0.1 mag)!')
    zp = 25
    if mags:
        g = mag2flux(g,zp)
        i = mag2flux(i,zp)
        z = mag2flux(z,zp)
        y = mag2flux(y,zp)
    cz = 0.5; cy = 0.5; cp = 0
    fit = (cz*z + cy*y)*(g/i)**cp
    if mags:
        filt = -2.5*np.log10(fit) + zp
    return filt