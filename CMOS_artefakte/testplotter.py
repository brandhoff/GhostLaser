# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:01:12 2021

@author: Jonas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from lmfit import Model, Parameters
from lmfit.models import VoigtModel
from lmfit.models import GaussianModel
from scipy import stats
from matplotlib.lines import Line2D
from random import randint
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import scipy.constants as consts
import matplotlib as mpl
file = r"D:\uni\PPraktikum\OCT\GhostLaser-main\powerCurve\430nm_STS.spec"

data = pd.read_csv(file, sep=',')
#fig, ax1 = plt.subplots(1, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    
#data.plot(ax = ax1,x='wavelength', y='intensity', c ='blue', label='Spectrum')



fileImg = r"D:\uni\PPraktikum\OCT\GhostLaser-main\powerCurve\440nm_CEMOS.data"
data = pd.read_csv(fileImg, sep=',')
#data = data.drop(0)
data.drop(data.columns[0], axis=1, inplace=True)
print(data)

D = data.to_numpy()
D = np.transpose(D)
print(max(map(max, D)))
plt.imshow(D, interpolation='none')
plt.show()