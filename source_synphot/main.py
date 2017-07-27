# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from . import io
from . import passband
import pysynphot as S

def main(inargs=None):
    args = io.get_options(args=inargs)

    sourcepb, sourcepbzp = io.get_passband(args.sourcepb)
    if sourcepbzp is None:
        sourcepbzp = np.nan

    new_sourcepbzp = passband.get_pb_zpt(sourcepb, model_mag=0.)
    if sourcepbzp != new_sourcepbzp:
        delta_zp = sourcepbzp - new_sourcepbzp
        print(delta_zp, sourcepbzp, new_sourcepbzp)
        sourcepbzp = new_sourcepbzp

    source_spec = S.FlatSpectrum(3631, fluxunits='jy')
    source_spec.convert('flam')

    ob = S.Observation(source_spec, sourcepb)
    print(ob.effstim('abmag'))

    out = passband.synphot(source_spec, sourcepb, sourcepbzp)
    print(out)
