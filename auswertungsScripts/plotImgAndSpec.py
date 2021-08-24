# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:01:12 2021
@author: Jonas
"""
import scipy.optimize as opt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy  as sp
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
    print(np.sum(D))
    ax1.imshow(D, interpolation='none')
    plt.show()
    
    
def showImg(file):
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    dataImg = pd.read_csv(file, sep=',')        
    dataImg.drop(dataImg.columns[0], axis=1, inplace=True)
    D = dataImg.to_numpy()
    D = np.transpose(D)
    ax1.imshow(D, interpolation='none')
    ax2.imshow(ROIfilter(dataImg))
    plt.show()

def twoD_Gaussian( xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def ROIfilter(dataImg):
    dataImg.drop(dataImg.columns[0], axis=1, inplace=True)
    D = dataImg.to_numpy()
    D = np.transpose(D)
    roied = D[400:550, 600:750]
    #print(roied)
    return roied



def fitSpot(roied , init_gues = (50, 80, 60, 20, 40, 0, 10) ):
    fileImg = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\Pro1000\700nm_CEMOS.data"
    dataImg = pd.read_csv(fileImg, sep=',')        
    arr = roied

    x = np.linspace(0, len(arr[0,:]), len(arr[0,:]))
    y = np.linspace(0, len(arr[:,0]), len(arr[:,0]))
    x, y = np.meshgrid(x, y)

    
    initial_guess = init_gues

    data_noisy = arr.ravel()
   # print((x, y))
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess,maxfev = 10600)#5600
    data_fitted = twoD_Gaussian((x, y), *popt)
    

    fig, ax = plt.subplots(1, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))

    ax.imshow(data_noisy.reshape(len(arr[:,0]), len(arr[0,:])))
    ax.contour(x, y, data_fitted.reshape(len(arr[0,:]), len(arr[:,0])), 8, colors='w')
    plt.show()
    #integrate the plot
    #print(max(data_fitted))#.reshape(len(arr[0,:]), len(arr[:,0]))))
    return np.sum(data_fitted)
    
    
    
    
    

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
   # fig.savefig("PowerCurve_fineSweep_10step.pdf");

    

def plotCurveUsingMaxima():
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
        intensities.append(max(map(max, D)))
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
    fig.savefig("PowerCurve_UsingMaxima.pdf");
    

def createNormalisationCurve():
    startLength = 430
    endLength = 700
    step = 1
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


    maxInte = max(intensities)
    caliIntensities = intensities/maxInte
    print(caliIntensities)
    ax1.plot(wavelengths,caliIntensities)
    
    
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
    return wavelengths, caliIntensities





def createSTSnormalisation():
    startLength = 430
    endLength = 700
    step = 1
    fig, (ax1,ax2) = plt.subplots(2, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    intensities = []
    wavelengths = []
    for i in range(startLength, endLength+step, step):
        file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\\"+str(i)+"nm_STS.spec"
        dataSpec = pd.read_csv(file, sep=',')
        ints = dataSpec['intensity'].values
        intensities.append(max(ints))
        wavelengths.append(i)


    maxInte = max(intensities)
    caliIntensities = intensities/maxInte
    print(caliIntensities)
    ax1.plot(wavelengths,caliIntensities)
    
    
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
    
    ax2.set_xlim(420,710)
    ax1.set_xlim(420,710)
    plt.show()
    df = pd.DataFrame()
    df.insert(0,"wavelength",wavelengths)
    df.insert(0,"intensity",caliIntensities)
    df.to_csv("STSCalibration_zweite.spec")

    
r"""   
def plotPatternWithInterpol(startLength, endLength, step):
    fig, ax = plt.subplots(1, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    intensities = []
    wavelengths = []
    dataCalibration = pd.read_csv('IntensityCalibration.spec', sep=',')
    
    dataCalibration = dataCalibration[dataCalibration.wavelength >= 500]
    
    caliIntensities = dataCalibration['intensity'].values
    caliWavelengths = dataCalibration['wavelength'].values
    
    intepolData = pd.read_csv( r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\sinePattern\\500nm_STS.spec", sep=',')
    intepolData = intepolData[intepolData.wavelength >= 500]
    intepolData = intepolData[intepolData.wavelength <= 700]
    interPolWave = intepolData['wavelength'].values
    
    new_x = np.linspace(startLength, endLength, len(interPolWave))
    calibration = sp.interpolate.interp1d(caliWavelengths, caliIntensities, kind='cubic')(new_x)
    
    print(caliIntensities)
    for i in range(startLength, endLength+step, step):
        file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\sinePattern\\"+str(i)+"nm_STS.spec"
        dataSpec = pd.read_csv(file, sep=',')
        dataSpec = dataSpec[dataSpec.wavelength >= 500]
        dataSpec = dataSpec[dataSpec.wavelength <= 700]
        x= dataSpec['wavelength'].values
        y = dataSpec['intensity'].values / calibration
        ax.plot(x,y, label='Spectrum')
        


   
    
    

    
    ax.tick_params(axis="x",direction="in")
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(labelleft=True)
    ax.set_xlabel('Wavelength', fontsize=16)
    ax.set_ylabel('Intensity', fontsize=16)
    
    ax.get_legend().remove()
    ax.set_xlim(420,710)
    ax.set_xlim(420,710)
    plt.show()
    fig.savefig("SinePattern.pdf");

"""   


def plotPattern(startLength, endLength, step):
    fig, ax = plt.subplots(1, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    intensities = []
    wavelengths = []
    dataCalibration = pd.read_csv('LaserSTSCalibration_merged.spec', sep=',')
    
    dataCalibration = dataCalibration[dataCalibration.wavelength >= startLength]
    dataCalibration = dataCalibration[dataCalibration.wavelength <= endLength]
    caliIntensities = dataCalibration['intensity'].values
    caliWavelengths = dataCalibration['wavelength'].values
    print(caliIntensities)
    
    for i in range(startLength, endLength+step, step):
        file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\sinePattern\\"+str(i)+"nm_STS.spec"
        dataSpec = pd.read_csv(file, sep=',')
        dataSpec = dataSpec[dataSpec.wavelength >= startLength]
        dataSpec = dataSpec[dataSpec.wavelength <= endLength]
        x= dataSpec['wavelength'].values
        y = dataSpec['intensity'].values
        
        wavelengths.append(i)
        intensities.append(max(y))
        
        
        


   
    

    normInt = intensities#/caliIntensities
    ax.plot(wavelengths,normInt, label='Spectrum')
    ax.tick_params(axis="x",direction="in")
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(labelleft=True)
    ax.set_xlabel('Wavelength', fontsize=16)
    ax.set_ylabel('Intensity', fontsize=16)
    
    ax.set_xlim(420,710)
    ax.set_xlim(420,710)
    plt.show()
    #fig.savefig("SinePattern.pdf");



def createLaserConfig():
    dataCalibration = pd.read_csv('STSCalibration_new.spec', sep=',')
    caliIntensities = dataCalibration['intensity'].values
    caliWavelengths = dataCalibration['wavelength'].values
    IntMinimum = min(caliIntensities)
    newCaliInt = caliIntensities * (1/IntMinimum)
    df = pd.DataFrame()
    df.insert(0,"wavelength",caliWavelengths)
    df.insert(0,"intensity",newCaliInt)
    df.to_csv("LaserSTSCalibration_new.spec")
    
    dataCalibration = pd.read_csv('IntensityCalibration.spec', sep=',')
    caliIntensities = dataCalibration['intensity'].values
    caliWavelengths = dataCalibration['wavelength'].values
    IntMinimum = min(caliIntensities)
    newCaliInt = caliIntensities * (1/IntMinimum)
    df = pd.DataFrame()
    df.insert(0,"wavelength",caliWavelengths)
    df.insert(0,"intensity",newCaliInt)
    df.to_csv("LaserCMOSCalibration.spec")
    
    
def mergeCalibrations():
    dataCalibration1 = pd.read_csv('STSCalibration_new.spec', sep=',')
    caliIntensities = dataCalibration1['intensity'].values
    caliWavelengths = dataCalibration1['wavelength'].values
    dataCalibration2 = pd.read_csv('STSCalibration_zweite.spec', sep=',')
    caliIntensities2 = dataCalibration1['intensity'].values
    
    mergedInts = caliIntensities * caliIntensities2
    IntMinimum = min(mergedInts)
    newCaliInt = mergedInts * (1/IntMinimum)
    df = pd.DataFrame()
    df.insert(0,"wavelength",caliWavelengths)
    df.insert(0,"intensity",newCaliInt)
    df.to_csv("LaserSTSCalibration_merged.spec")
    
    
    
def plot2DCurve():
    startLength = 430
    endLength = 700
    step = 10
    startPromille = 0
    endPromille = 1000
    stepPromille = 100 
    intensities_list = []
    
    for j in range(startPromille, endPromille+stepPromille, stepPromille):
            fig, (ax1,ax2) = plt.subplots(2, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
            intensities = []
            wavelengths = []
            for i in range(startLength, endLength+step, step):
                fileImg = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve/Pro"+str(j)+"/"+str(i)+"nm_CEMOS.data"
                file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve/Pro"+str(j)+"/"+str(i)+"nm_STS.spec"
                dataSpec = pd.read_csv(file, sep=',')
                dataSpec.plot(ax = ax2,x='wavelength', y='intensity', c ='blue', label='Spectrum')
                dataImg = pd.read_csv(fileImg, sep=',')
                D = ROIfilter(dataImg)
                fitres = fitSpot(D, init_gues = (50, 80, 60, 20, 40, 0, 10))
                if j == 0 :
                    fitres = fitSpot(D, init_gues = (0, 80, 60, 20, 40, 0, 10))
                intensities.append(fitres)
                wavelengths.append(i)

            ax1.plot(wavelengths,intensities)
    
    
            ax1.tick_params(axis="x",direction="in")
            ax1.tick_params(axis="y",direction="in")
        
            ax1.tick_params(labelleft=True)
            ax1.set_xlabel('Satz promille'+str(j), fontsize=16)
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
            intensities_list.append(intensities)
    return np.array(intensities_list)
            
def plotWave(waeset):
    wave = waeset
    startPromille = 0
    endPromille = 1000
    stepPromille = 100 
    fig, (ax1,ax2) = plt.subplots(2, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
    intensities = []
    promilles = []
    for j in range(startPromille, endPromille+stepPromille, stepPromille):

                fileImg = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve/Pro"+str(j)+"/"+str(wave)+"nm_CEMOS.data"
                file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve/Pro"+str(j)+"/"+str(wave)+"nm_STS.spec"
                dataSpec = pd.read_csv(file, sep=',')
                dataSpec.plot(ax = ax2,x='wavelength', y='intensity', c ='blue', label='Spectrum')
                dataImg = pd.read_csv(fileImg, sep=',')
        
                dataImg.drop(dataImg.columns[0], axis=1, inplace=True)
                promilles.append(j)
                D = dataImg.to_numpy()
                intensities.append(np.sum(D))

    ax1.plot(promilles,intensities)
    ax1.tick_params(axis="x",direction="in")
    ax1.tick_params(axis="y",direction="in")
    ax1.tick_params(labelleft=True)
    ax1.set_xlabel('Satz promille'+str(j), fontsize=16)
    ax1.set_ylabel('Intensity', fontsize=16)
        
    ax2.tick_params(axis="x",direction="in")
    ax2.tick_params(axis="y",direction="in")
    ax2.tick_params(labelleft=True)
    ax2.set_xlabel('Wavelength', fontsize=16)
    ax2.set_ylabel('Intensity', fontsize=16)
        
    ax2.get_legend().remove()

    plt.show()
            
            
def createMatrix():
    startLength = 430
    endLength = 700
    step = 1
    startPromille = 0
    endPromille = 1000
    stepPromille = 100 
    matrix = np.zeros((10, 270))
    for j in range(startPromille, endPromille+stepPromille, stepPromille):
            for i in range(startLength, endLength+step, step):
                fileImg = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve/Pro"+str(j)+"/"+str(i)+"nm_CEMOS.data"
                file = r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve/Pro"+str(j)+"/"+str(i)+"nm_STS.spec"
                dataSpec = pd.read_csv(file, sep=',')
                dataImg = pd.read_csv(fileImg, sep=',')
        
                dataImg.drop(dataImg.columns[0], axis=1, inplace=True)
       
                D = dataImg.to_numpy()
                matrix[i][j]=np.sum(D)

    print(matrix)
#createLaserConfig()   
#plotPattern(430, 700, 1)
#plotCurve()
#plotWave(441)
#createMatrix()
#createSTSnormalisation()
#mergeCalibrations()
"""
wavelengths, caliIntensities = createNormalisationCurve()
df = pd.DataFrame()
df.insert(0,"wavelength",wavelengths)
df.insert(0,"intensity",caliIntensities)
df.to_csv("IntensityCalibration")
"""
#singlePlot("1629462058")
#singlePlot("1629462196")
#singlePlot("1629462347")
#showImg(r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\Pro100\700nm_CEMOS.data")
#fitSpot("")
#singlePlot("1629462362")
#ROIfilter("")
intensities = plot2DCurve()

np.savetxt(r"C:\Users\Labor_NLO-Admin\Desktop\GhostLaser\scripts\powerCurve\matrix_flipped.txt",intensities)

plt.pcolormesh(intensities,cmap="jet")
plt.ylabel("intensity [promille]")
plt.xlabel("wavelength")
plt.show()

plt.title(str(430+28*10)+"nm")
plt.plot(intensities[:,27])
plt.xlabel("laser intensity")
plt.ylabel("measured intensity")
plt.show()