# -*- coding: UTF-8 -*-
"""
Entry point for source_synphot package
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import sys
from .main import main

if __name__=='__main__':
    inargs = sys.argv[1:]
    main(inargs)
