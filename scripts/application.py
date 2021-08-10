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

from MainWindow import Ui_MainWindow

import numpy as np

import asyncio
from PyQt5.QtCore import QTimer
from asyncqt import QEventLoop, asyncSlot
import IPython
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
       self.btnRandomPattern.clicked.connect(self.startRandomPattern)
    
    
    
    
    
    
    def toggleEmission(self):
        if self.EmissionOn:
            self.laser.EmissionOff()
            self.EmissionOn = False
            self.btnToggleEmission.setStyleSheet("border: 3px solid rgba(245,0,0,255);\n""border-radius: 10px;\n""background-color: rgba(245,0,0,255);")
            
            
        else:
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
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = Window()
    win.show()
    
    with loop:
        loop.run_forever()

    sys.exit(app.exec())