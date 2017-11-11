#!/usr/local/bin/python
'''
*
*  written by Guido Novati: novatig@ethz.ch
*  Copyright 2017 ETH Zurich. All rights reserved.
*
'''
import os, pathlib, sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

p = pathlib.Path('./')
for comp in list(p.glob('component_*.raw')):
    D = np.fromfile(comp.name, dtype=np.float32)
    D.resize([28, 28])
    plt.imshow(D)
    plt.show()
