# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import os
import numpy as np
from . import io
from . import passband

def main(inargs=None):
    pbzptfile = os.path.join('passbands','pbzptmag.txt')
    pbzptfile = io.get_pkgfile(pbzptfile)
    pbzpt = np.recfromtxt(pbzptfile, names=True)
    pbnames  = pbzpt.obsmode
    pbnames = [x.decode('latin-1') for x in pbnames]

    pbs = passband.load_pbs(pbnames, 0.)
    #print(pbs)
