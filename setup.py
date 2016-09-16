#! /usr/bin/env python

from setuptools import setup

setup(name='SODA',
      version='0.1',
      description='Satellite Orbits DynAmics',
      author='Nicolas Garavito',
      author_email='jngaravitoc@email.arizona.edu',
      install_requieres=['numpy', 'scipy', 'astropy'],
      packages=['soda'],
     )
