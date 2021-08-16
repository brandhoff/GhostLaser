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
import IPython
from seabreeze.spectrometers import Spectrometer as sm
#IPython.get_ipython().run_line_magic('matplotlib', 'qt')

def singlePlot(prefix):
    file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\imgs\\"+prefix+"_SPEC.spec"
    fileImg = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\imgs\\"+prefix+"_CEMOS.data"
    dataSpec = pd.read_csv(file, sep=',')
    fig, (ax1,ax2) = plt.subplots(2, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    dataSpec.plot(ax = ax2,x='wavelength', y='intensity', c ='blue', label='Spectrum')   
    dataImg = pd.read_csv(fileImg, sep=',')        
    dataImg.drop(dataImg.columns[0], axis=1, inplace=True)
    print(dataImg)    
    D = dataImg.to_numpy()
    D = np.transpose(D)
    print(max(map(max, D)))
    ax1.imshow(D, interpolation='none')
    plt.show()


def multiPlot():
    startLength = 430
    endLength = 700
    step = 10
    for i in range(startLength, endLength+step, step):
    
    
    
    
        fileImg = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\\"+str(i)+"nm_CEMOS.data"
        file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\\"+str(i)+"nm_STS.spec"
        dataSpec = pd.read_csv(file, sep=',')
        fig, (ax1,ax2) = plt.subplots(2, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
            
        dataSpec.plot(ax = ax2,x='wavelength', y='intensity', c ='blue', label='Spectrum')
        
        
        
        dataImg = pd.read_csv(fileImg, sep=',')
        
        
        dataImg.drop(dataImg.columns[0], axis=1, inplace=True)
        print(dataImg)
        
        D = dataImg.to_numpy()
        D = np.transpose(D)
        print(max(map(max, D)))
        ax1.imshow(D, interpolation='none')
        plt.show()
        
#singlePlot("1629104036")
#multiPlot()


def plotCurve():
    startLength = 430
    endLength = 700
    step = 10
    fig, (ax1,ax2) = plt.subplots(2, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    intensities = []
    wavelengths = []
    for i in range(startLength, endLength+step, step):
        fileImg = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\\"+str(i)+"nm_CEMOS.data"
        file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\\"+str(i)+"nm_STS.spec"
        dataSpec = pd.read_csv(file, sep=',')
        dataSpec.plot(ax = ax2,x='wavelength', y='intensity', c ='blue', label='Spectrum')
        dataImg = pd.read_csv(fileImg, sep=',')
        
        dataImg.drop(dataImg.columns[0], axis=1, inplace=True)
       
        D = dataImg.to_numpy()
        intensities.append(np.sum(D))
        wavelengths.append(i)

    ax1.plot(wavelengths,intensities)
    
    
    ax1.tick_params(axis="x",direction="in")
    ax1.tick_params(axis="y",direction="in")
    
    ax1.tick_params(labelleft=True)
    ax1.set_xlabel('Wavelength', fontsize=16)
    ax1.set_ylabel('Intensity', fontsize=16)
    
    ax2.tick_params(axis="x",direction="in")
    ax2.tick_params(axis="y",direction="in")
    ax2.tick_params(labelleft=True)
    ax2.set_xlabel('Wavelength', fontsize=16)
    ax2.set_ylabel('Intensity', fontsize=16)
    
    ax2.get_legend().remove()
    ax2.set_xlim(420,710)
    ax1.set_xlim(420,710)
    plt.show()
    fig.savefig("PowerCurve.pdf");
    
plotCurve()


