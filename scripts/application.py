#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:09:21 2021

@author: Jonas Brandhoff
"""
import os
import sys
from NKTP_DLL import *
import Laser
import random
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QFileDialog
from datetime import datetime
from MainWindow import Ui_MainWindow
import pandas as pd
import numpy as np
import time
import asyncio
from PyQt5.QtCore import QTimer
from asyncqt import QEventLoop, asyncSlot
import Camera.camera as ca
import IPython
import Pattern as pt
from seabreeze.spectrometers import Spectrometer as sm
import matplotlib.pyplot as plt
IPython.get_ipython().run_line_magic('matplotlib', 'qt')


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalsSlots()
        self.EmissionOn = False
        self.RFOn = False
        self.laser = Laser.Laser()
        self.laser.openConnection()
        
        self.timer=QTimer()
        self.runningTimer = False
        self.cam = ca.Camera()
        self.spec = sm.from_first_available()
        self.PatternHelper = pt.PatternHelper()
        
        
        dataCalibration = pd.read_csv('Calibrations/LaserSTSCalibration_merged.spec', sep=',')
        self.caliSTSint = dataCalibration['intensity'].values
        self.caliSTSwave = dataCalibration['wavelength'].values
        
        dataCalibration = pd.read_csv('Calibrations/LaserCMOSCalibration.spec', sep=',')
        self.caliCMOSint = dataCalibration['intensity'].values
        self.caliCMOSwave = dataCalibration['wavelength'].values

    """
    HIER WERDEN DIE EINZELNEN SIGNALS MIT DEN SLOTS VERBUNDEN
    siehe QT5 Doc
    """
    def connectSignalsSlots(self):
       self.btnExit.clicked.connect(self.exitOut)
       
       """
    
    LASER zeug
    
    """
       self.btnToggleEmission.clicked.connect(self.toggleEmission)
       self.btnToggleRF.clicked.connect(self.toggleRF)
       self.btnSetOutput.clicked.connect(self.changeLaser)
       self.btnRandomPattern.clicked.connect(self.fastSweepThrouPattern)
       """
       Power curve
       
       """
       self.btnTakeCurve.clicked.connect(self.Take2DCurve)
       
       
       """
       Camera stuff
       """
       self.pushButton.clicked.connect(self.camerShot)
    
    
    def toggleEmission(self):
        if self.EmissionOn:
            self.laser.EmissionOff()
            self.EmissionOn = False
            self.btnToggleEmission.setStyleSheet("border: 3px solid rgba(245,0,0,255);\n""border-radius: 10px;\n""background-color: rgba(245,0,0,255);")
            
            
        else:
            self.laser.resetInterLock()
            self.laser.EmissionOn()
            self.EmissionOn = True
            self.btnToggleEmission.setStyleSheet("border: 3px solid rgba(51, 181, 25,255);\n""border-radius: 10px;\n""background-color: rgba(51, 181, 25,255);")

    def toggleRF(self):
        if self.RFOn:
            self.laser.RFOff()
            self.RFOn = False
            self.btnToggleRF.setStyleSheet("border: 3px solid rgba(245,0,0,255);\n""border-radius: 10px;\n""background-color: rgba(245,0,0,255);")
            
            
        else:
            self.laser.RFOn()
            self.RFOn = True
            self.btnToggleRF.setStyleSheet("border: 3px solid rgba(51, 181, 25,255);\n""border-radius: 10px;\n""background-color: rgba(51, 181, 25,255);")

    

    def changeLaser(self):
        wav = self.spinWave.value()
        num = self.spinLaserNum.value()
        amp = self.spinAmp.value()
        self.laser.setLaserOutputPM(num, wav, amp)


 
        
    def exitOut(self):
       
        self.laser.RFOff()
        self.laser.EmissionOff()
        self.laser.closeConnection()
        self.cam.dispose()
        self.spec.close()
        os._exit(0)
        self.close()
        loop.stop()
        
    
    
    """
    --------------------------------------------------------------------------------------------------------
    --------------------------------GHOST PATTERNS----------------------------------------------------------
    --------------------------------------------------------------------------------------------------------
    """
    def startRandomPattern(self):
        
        if self.runningTimer:
            self.timer.stop()
            self.runningTimer = False
            self.timer.timeout.disconnect()
            return
        
        print("RANDOM PATTERN STARTED")
        self.timer.timeout.connect(self.doRandomPattern)
        self.runningTimer = True
        self.timer.start(1)
    def doRandomPattern(self):
        for i in range(8):
            wavelength = random.randint(4300,7000)
            amplitude = random.randint(1,101)
            self.laser.setLaserOutputPM(i, wavelength*100, amplitude*10)
    
    
    
    
    def imageToCSV(self, image, filename):
        df = pd.DataFrame()
        for i, im in enumerate(image):
            df.insert(0,"image"+str(i),im)
        df.to_csv(filename)
    
    def stsToCSV(self, intensity, wave, filename):
        df = pd.DataFrame()
        df.insert(0,"wavelength",wave)
        df.insert(0,"intensity",intensity)
        df.to_csv(filename)
    
    def getIndexForWave(self, wavelenght):
        exists = wavelenght in self.caliSTSwave
        if exists:
            index = list(self.caliSTSwave).index(wavelenght)
            return index
        else:
            print("ERROR WAVELENGTH NOT PRESENT RETURNING 0")
            return 0
    
    def getCalibratedIntensityFactor(self, wavelenght, STS):
        extraFunction = 1.5
        if STS:
            return 1/(extraFunction*self.caliSTSint[self.getIndexForWave(wavelenght)])
        else:
            return 1/self.caliCMOSint[self.getIndexForWave(wavelenght)]
    
    
    @asyncSlot()
    async def fastSweepThrouPattern(self):
        pattern = self.PatternHelper.getSineIntegerLayout(430,700,1, 500, 900, 0,0* 1/(730-430)*8*np.pi)
        plotWave = []
        plotInt = []
        self.spec.integration_time_micros(2000)
        for i in range(len(pattern.intensities)):
            self.laser.setLaserOutputPM(0, pattern.wavelengths[i]*1000, int(pattern.intensities[i]*self.getCalibratedIntensityFactor(pattern.wavelengths[i],True)))
            plotWave.append(pattern.wavelengths[i])
            plotInt.append(int(pattern.intensities[i]*self.getCalibratedIntensityFactor(pattern.wavelengths[i],True)))
            current_wave = self.spec.wavelengths()
            current_int = self.spec.intensities()
            self.stsToCSV(current_int,current_wave,"sinePattern/"+str(pattern.wavelengths[i])+"nm_STS.spec")
            schritteGes = len(pattern.wavelengths)
            momSchritt = i+1
            percent = momSchritt/schritteGes
            print(percent)
            self.progressCurve.setValue(percent*100)
        fig, ax = plt.subplots(1, gridspec_kw={'wspace': 0}, facecolor='w',figsize=(11,8))
        ax.plot(plotWave,plotInt, label='Spectrum')

    """
    --------------------------------------------------------------------------------------------------------
    --------------------------------Take Curve----------------------------------------------------------
    --------------------------------------------------------------------------------------------------------
    """
    @asyncSlot()
    async def startTakeCurve(self):
        startLength = 430
        endLength = 700
        step = 1
        self.spec.integration_time_micros(2000)
        self.cam.setExposure(0)
        self.cam.setTimeOut(100)
        
        
        for i in range(startLength, endLength+step, step):
            self.laser.setLaserOutputPM(0, i*1000, int(10000*self.getCalibratedIntensityFactor(i,True)))
            img = self.cam.takeImage()
            current_wave = self.spec.wavelengths()
            current_int = self.spec.intensities()
            self.imageToCSV(img, "powerCurve/"+str(i)+"nm_CEMOS.data")
            self.stsToCSV(current_int,current_wave,"powerCurve/"+str(i)+"nm_STS.spec")
            schritteGes = (endLength-startLength)/step
            momSchritt = (i - startLength)/step
            percent = momSchritt/schritteGes
            print(percent)
            self.progressCurve.setValue(percent*100)
        
    @asyncSlot()
    async def Take2DCurve(self):
        startLength = 430
        endLength = 700
        step = 1
        self.spec.integration_time_micros(2000)
        self.cam.setExposure(0)
        self.cam.setTimeOut(100)
        startPromille = 0
        endPromille = 1000
        stepPromille = 100
        # Das sind insgesamt 5400 dateien! manuell geht da nichts mehr!
        #zum handeln einfach die zwei for loops kopieren und die dateien einlesen von den Pfaden
        # geschaetzte gesamt groesse fuer die matrix wird 8.2GB !!!
        #daher bitte nicht mehr stuezstellen aufnehmen und wenn dann interpolieren
        for j in range(startPromille, endPromille+stepPromille, stepPromille):
            for i in range(startLength, endLength+step, step):
                self.laser.setLaserOutputPM(0, i*1000, int(10000*self.getCalibratedIntensityFactor(i,True)))
                img = self.cam.takeImage()
                current_wave = self.spec.wavelengths()
                current_int = self.spec.intensities()
                self.imageToCSV(img, "powerCurve/Pro"+str(j)+"/"+str(i)+"nm_CEMOS.data")
                self.stsToCSV(current_int,current_wave,"powerCurve/Pro"+str(j)+"/"+str(i)+"nm_STS.spec")
            schritteGes = (endPromille-startPromille)/stepPromille
            momSchritt = (j - startPromille)/stepPromille
            percent = momSchritt/schritteGes
            print(percent)
            self.progressCurve.setValue(percent*100)
        
        
    @asyncSlot()
    async def camerShot(self):
            now = round(time.time())
            self.cam.setExposure(5)
            img = self.cam.takeImage()
            current_wave = self.spec.wavelengths()
            current_int = self.spec.intensities()
            self.imageToCSV(img, "imgs/"+str(now)+"_CEMOS.data")
            self.stsToCSV(current_int,current_wave,"imgs/"+str(now)+"_SPEC.spec")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = Window()
    win.show()
    
    #with loop:
        #loop.run_forever()

    sys.exit(app.exec())