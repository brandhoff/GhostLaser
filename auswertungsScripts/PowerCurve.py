# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:42:44 2021

@author: Labor_NLO-Admin
"""

import pandas as pd
import numpy as np


filepath = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\\"



startLength = 430
endLength = 700
step = 10
for i in range(startLength, endLength+step, step):
    filePRefix = filepath+str(i)+"nm_STS.spec"
    data = pd.read_csv(filePRefix, sep=',')