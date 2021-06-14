#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = ['convex_covering_spaces', 'data_structures', 'delaunay_triangulation', 'quasi_delaunay_triangulation']

setup(name='pyQDT',
      version='0.1',
      author='Dmitry Yershov',
      author_email='dmitry.s.yershov@gmail.com',
      description='Quasi Delaunay Triangulation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      package_dir = {'': 'src'},
      packages=packages,
      install_requires=['numpy'],
      extras_require={'interactive':['ipython',
                                     'jupyter',
                                     'interactive-plotter @ git+https://github.com/dyershov/interactive-plotter#egg=interactive-plotter',
                                     ],
                      },
     )
