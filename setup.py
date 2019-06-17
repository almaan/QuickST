#!/usr/bin/env python3

from setuptools import setup
import os

setup(name='QuickST',
            version='0.1',
            description='For easy handling of ST-data',
            url='http://github.com/almaan/QuickST',
            author='Alma Andersson',
            author_email='alma.andersson@scilifelab.se',
            license='MIT',
            packages=['QuickST',
                     'QuickST.data',
                     'QuickST.visual'],
            install_requires=[
                            'numpy',
                            'pandas',
                            'scipy',
                            'logging',
                            'argparse',
                            'matplotlib',
                            'sklearn',
                            'umap-learn',
                      ],
            zip_safe=False)



