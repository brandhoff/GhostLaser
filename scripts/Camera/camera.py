# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:51:41 2021

@author: Labor_NLO-Admin
"""
try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE


class Camera:
    
    
    def __init__(self):
        return
    
    def openCamera(self):
        print("yoo")
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")
            self.cameraInstance = sdk.open_camera(available_cameras[0])
           
            print(self.cameraInstance)
            
           
    def setExposure(self, timeMS):
         with TLCameraSDK() as sdk:
             print(self.cameraInstance)
             self.cameraInstance.exposure_time_us = timeMS*1000
             
"""            
    def setExposure(self, timeMS):
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")
            with sdk.open_camera(available_cameras[0]) as camera:
                camera.frames_per_trigger_zero_for_unlimited = 0
                camera.exposure_time_us = timeMS*1000
        
    def setTimeOut(self, timeOutMS):
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")
            with sdk.open_camera(available_cameras[0]) as camera:
                camera.image_poll_timeout_ms = timeOutMS

        
        
 
                
        
    
    def takeImage(self):      
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")
        
            with sdk.open_camera(available_cameras[0]) as camera:

                old_roi = camera.roi  # store the current roi
                    
                camera.arm(2)
        
                camera.issue_software_trigger()
                frame = camera.get_pending_frame_or_null()  
                image_buffer_copy = None
                if frame is not None:       
                        frame.image_buffer 
                        image_buffer_copy = np.copy(frame.image_buffer)
                else:
                        print("timeout reached during polling, program exiting...")
                camera.disarm()
                return(image_buffer_copy)

        
    
"""     
cam = Camera()
cam.openCamera()
cam.setExposure(10)

