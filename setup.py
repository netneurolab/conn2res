#!/usr/bin/env python

import versioneer
import os
from setuptools import setup
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    setup(name='conn2res',
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass())
