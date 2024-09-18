# -*- coding: utf-8 -*-
"""
Setup script for the WDmodel package
"""
import os
import re
import glob

from setuptools import find_packages, setup

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'source_synphot', '__init__.py')).read()
VERS = r"^__version__\s+=\s+[\'\"]([0-9\.]*)[\'\"]$"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)
AUTH = r"^__author__\s+=\s+[\'\"]([A-za-z\s]*)[\'\"]$"
mo = re.search(AUTH, init_string, re.M)
__author__ = mo.group(1)
LICE = r"^__license__ \s+=\s+[\'\"]([A-za-z\s0-9]*)[\'\"]$"
mo = re.search(LICE, init_string, re.M)
__license__ = mo.group(1)

long_description = open('README.rst').read()

scripts = glob.glob('bin/*')
print(scripts)

setup(
    name='source_synphot',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'source_synphot = source_synphot.main:main'
    ]},
    include_package_data=True,
    version=__version__,  # noqa
    description=('Synthetic photometry of arbitrary sources for JWST/NIRCam and WFIRST'),
    scripts = scripts,
    license=__license__,  # noqa
    author=__author__,  # noqa
    author_email='gsnarayan@gmail.com',
    install_requires=required,
    url='https://github.com/gnarayan/source_synphot',
    keywords=['astronomy', 'photometry', 'calibration', 'JWST', 'WFRIST', 'synethetic'],
    long_description=long_description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: GNU Public License v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics'
    ])
