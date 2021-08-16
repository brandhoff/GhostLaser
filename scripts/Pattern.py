# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:18:10 2021

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
import IPython
IPython.get_ipython().run_line_magic('matplotlib', 'qt')


class Pattern():
    def __init__(self,wavelengths,intensities):
        self.wavelengths = wavelengths
        self.intensities = intensities
        
    def plot(self):
        fig, ax1 = plt.subplots(1, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
        ax1.plot(self.wavelengths,self.intensities)

    def getIntForWave(self, wavelenght):
        exists = wavelenght in self.wavelengths
        if exists:
            index = list(self.wavelengths).index(wavelenght)
            return self.intensities[index]
        else:
            print("ERROR WAVELENGTH NOT PRESENT RETURNING 0")
            return 0
        
        
        
class PatternHelper():
    def __init__(self):
        return
    
    def getSineLayout(self, startWave, endWave, step, amplitude, yOffset, phase, frequ):
        area = list(range(startWave, endWave+step, step))
        
        ar = np.array(area)
        function = amplitude*np.sin(ar*frequ+phase)+yOffset
        return Pattern(ar, function)

        
    def getSineIntegerLayout(self, startWave, endWave, step, amplitude, yOffset, phase, frequ):
        area = list(range(startWave, endWave+step, step))
        
        ar = np.array(area)
        function = amplitude*np.sin(ar*frequ+phase)+yOffset
        function = np.floor(function)
        function = function.astype(int)
        return Pattern(ar, function)
    
    
    def getCosIntegerLayout(self, startWave, endWave, step, amplitude, yOffset, phase, frequ):
        area = list(range(startWave, endWave+step, step))
        ar = np.array(area)
        function = amplitude*np.cos(ar*frequ+phase)+yOffset
        function = np.floor(function)
        function = function.astype(int)
        return Pattern(ar, function)
    
    def getRandomIntegerLayout(self, startWave, endWave, step, mini, maxi):
        area = list(range(startWave, endWave+step, step))
        ar = np.array(area)
        function = np.random.randint(mini,maxi,len(area))
        return Pattern(ar, function)

        
p = PatternHelper()
pattern = p.getSineIntegerLayout(430,700,1, 500, 500, 0, 1/(700-430)*2*np.pi)
pattern.plot()
print(pattern.getIntForWave(500))
patterncos = p.getCosIntegerLayout(430,700,1, 500, 500, 0, 1/(700-430)*2*np.pi)
patterncos.plot()
print(patterncos.getIntForWave(500))

patternRand = p.getRandomIntegerLayout(430,700,1, 0, 1000)
patternRand.plot()
print(patternRand.getIntForWave(500))
