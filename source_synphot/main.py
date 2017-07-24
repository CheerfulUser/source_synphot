# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from . import io

def main(inargs=None):
    args = io.get_options(args=inargs)
    print(args)
