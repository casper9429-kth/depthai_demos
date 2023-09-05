# Does not work good
import cv2
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
from PIL import Image, ImageTk
import depthai as dai

class DepthCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Camera App")

        # Create a canvas to display the camera feed
        self.canvas = tk.Canvas(self.root, width=640*2, height=480)
        self.canvas.pack()

        # Create buttons
        ## Median filter
        self.median_filter_list = [dai.MedianFilter.MEDIAN_OFF, dai.MedianFilter.KERNEL_3x3, dai.MedianFilter.KERNEL_5x5, dai.MedianFilter.KERNEL_7x7]
        self.median_filter_list_str = ["Median Filter Off", "Median Filter 3x3", "Median Filter 5x5", "Median Filter 7x7"]
        self.median_filter_list_index = 0
        ## Subpixel
        self.subpixel_list = [False, True]
        self.subpixel_list_str = ["Subpixel Off", "Subpixel On"]
        self.subpixel_list_index = 0
        ## LR check
        self.lr_check_list = [False, True]
        self.lr_check_list_str = ["LR Check Off", "LR Check On"]
        self.lr_check_list_index = 0
        ## Extended disparity
        self.extended_disparity_list = [False, True]
        self.extended_disparity_list_str = ["Extended Disparity Off", "Extended Disparity On"]
        self.extended_disparity_list_index = 0
        ## Speckle filter
        self.speckle_filter_list = [False, True]
        self.speckle_filter_list_str = ["Speckle Filter Off", "Speckle Filter On"]
        self.speckle_filter_list_index = 0
        ## Speckle Value
        self.speckle_value = 50
        self.speckle_value_str = "Speckle Value: " + str(self.speckle_value)
        ## Confidence threshold
        self.confidence_threshold = 0
        self.confidence_threshold_str = "Confidence Threshold: " + str(self.confidence_threshold)
        ## Temporal Filter
        self.temporal_filter_list = [False, True]
        self.temporal_filter_list_str = ["Temporal Filter Off", "Temporal Filter On"]
        self.temporal_filter_list_index = 0

        ## Median filter
        self.median_filter_off = ttk.Button(self.root, text=self.median_filter_list_str[self.median_filter_list_index], command=self.update_median_filter) 
        ## Subpixel
        self.subpixel_off = ttk.Button(self.root, text=self.subpixel_list_str[self.subpixel_list_index], command=self.update_subpixel)
        ## LR check
        self.lr_check_off = ttk.Button(self.root, text=self.lr_check_list_str[self.lr_check_list_index], command=self.update_lr_check)
        ## Extended disparity
        self.extended_disparity_off = ttk.Button(self.root, text=self.extended_disparity_list_str[self.extended_disparity_list_index], command=self.update_extended_disparity)
        ## Start button
        self.start_button = ttk.Button(self.root, text="Start", command=self.start_camera)
        ## Confidence threshold
        self.confidence_threshold_off = ttk.Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_confidence_threshold)
        ## Speckle filter button on the side 
        self.speckle_filter_off = ttk.Button(self.root, text=self.speckle_filter_list_str[self.speckle_filter_list_index], command=self.update_speckle_filter)
        ## Speckle value
        self.speckle_value_off = ttk.Scale(self.root, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_speckle_value)
        ## Temporal filter
        self.temporal_filter_off = ttk.Button(self.root, text=self.temporal_filter_list_str[self.temporal_filter_list_index], command=self.update_temporal_filter)

        self.median_filter_off.pack()
        self.subpixel_off.pack()
        self.lr_check_off.pack()
        self.extended_disparity_off.pack()
        self.confidence_threshold_off.pack()
        self.start_button.pack()
        self.speckle_filter_off.pack(side=tk.RIGHT)
        self.speckle_value_off.pack(side=tk.RIGHT)
        self.temporal_filter_off.pack(side=tk.RIGHT)
        # Create a thread to capture camera frames
        self.thread = None
        self.is_running = False

        # Create a pipeline object for your depth camera setup
        self.pipeline = dai.Pipeline()

        # Define sources
        ## Left mono camera
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        ## Right mono camera
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        # Define stereo node
        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setRuntimeModeSwitch(True)
        # Define outputs and inputs
        ## Disparity
        xout_disp = self.pipeline.create(dai.node.XLinkOut)
        xout_disp.setStreamName("disparity")
        xout_disp.input.setBlocking(False)
        ## Depth
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        xout_depth.input.setBlocking(False)
        ## Define input to stereo node
        xin_stereo = self.pipeline.create(dai.node.XLinkIn)
        xin_stereo.setStreamName("stereo_config")

        # Linking
        ## Link left mono camera to stereo left and right mono camera to stereo right
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        ## Link stereo disparity to disparity XLINK output
        stereo.disparity.link(xout_disp.input)
        ## Link stereo depth to depth XLINK output
        stereo.depth.link(xout_depth.input)
        ## Link stereo config to stereo node input
        xin_stereo.out.link(stereo.inputConfig)
    
    def update_temporal_filter(self):
        self.temporal_filter_list_index = (self.temporal_filter_list_index + 1) % len(self.temporal_filter_list)
        # Update button text
        self.temporal_filter_off.config(text=self.temporal_filter_list_str[self.temporal_filter_list_index])
        
    def update_speckle_value(self, value):
        # parse string float to int
        self.speckle_value = int(float(value))
        # Update button text
        self.speckle_value_str = "Speckle Value: " + str(self.speckle_value)
    
    def update_speckle_filter(self):
        self.speckle_filter_list_index = (self.speckle_filter_list_index + 1) % len(self.speckle_filter_list)
        # Update button text
        self.speckle_filter_off.config(text=self.speckle_filter_list_str[self.speckle_filter_list_index])
        
        
    def update_median_filter(self):
        self.median_filter_list_index = (self.median_filter_list_index + 1) % len(self.median_filter_list)
        # Update button text
        self.median_filter_off.config(text=self.median_filter_list_str[self.median_filter_list_index])
    
    def update_subpixel(self):
        self.subpixel_list_index = (self.subpixel_list_index + 1) % len(self.subpixel_list)
        # Update button text
        self.subpixel_off.config(text=self.subpixel_list_str[self.subpixel_list_index])
    
    def update_lr_check(self):
        self.lr_check_list_index = (self.lr_check_list_index + 1) % len(self.lr_check_list)
        # Update button text
        self.lr_check_off.config(text=self.lr_check_list_str[self.lr_check_list_index])
    
    def update_extended_disparity(self):
        self.extended_disparity_list_index = (self.extended_disparity_list_index + 1) % len(self.extended_disparity_list)
        # Update button text
        self.extended_disparity_off.config(text=self.extended_disparity_list_str[self.extended_disparity_list_index])
    
    def update_confidence_threshold(self, value):
        # parse string float to int
        self.confidence_threshold = int(float(value))
        # Update button text
        self.confidence_threshold_str = "Confidence Threshold: " + str(self.confidence_threshold)
        
    def start_camera(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.capture_frames)
            self.thread.start()

    def stop_camera(self):
        self.is_running = False
        if self.thread:
            self.thread.join()

    def capture_frames(self):
        # Initialize the depth camera device
        with dai.Device(self.pipeline) as device:
            # Get XLINK input/output queues
            q_stereo_config = device.getInputQueue("stereo_config")
            q_disp = device.getOutputQueue("disparity", maxSize=4, blocking=False)
            q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

            while self.is_running:
                # Configure the median filter
                cfg = dai.StereoDepthConfig()
                cfg.setMedianFilter(self.median_filter_list[self.median_filter_list_index])
                cfg.setLeftRightCheck(self.lr_check_list[self.lr_check_list_index])
                cfg.setExtendedDisparity(self.extended_disparity_list[self.extended_disparity_list_index])
                cfg.setSubpixel(self.subpixel_list[self.subpixel_list_index])
                cfg.setConfidenceThreshold(self.confidence_threshold)


                cfg.PostProcessing.SpeckleFilter.enable = self.speckle_filter_list[self.speckle_filter_list_index]
                cfg.PostProcessing.SpeckleFilter.speckleRange = self.speckle_value
                cfg.PostProcessing.TemporalFilter.enable = self.temporal_filter_list[self.temporal_filter_list_index]
                
                cfg.setBilateralFilterSigma(16)
                
                
                q_stereo_config.send(cfg)
                
                
                
                in_disp = q_disp.get()
                in_depth = q_depth.get()

                # Get data from disparity frame
                disp = in_disp.getFrame()
                # Get data from depth frame
                depth = in_depth.getFrame()
                depth = (255 * depth / depth.max()).astype(np.uint8)
                # Color code by jet colormap
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                # Crop disparity frame
                disp = disp[50:430, 50:590]

                # Normalize disparity frame
                disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
                # Convert disparity frame to 8-bit
                disp = disp.astype(np.uint8)

                # Display both disp and depth frames
                photo_disp = ImageTk.PhotoImage(image=Image.fromarray(disp))
                self.canvas.create_image(0, 0, image=photo_disp, anchor=tk.NW)
                self.canvas.photo_disp = photo_disp
                photo_depth = ImageTk.PhotoImage(image=Image.fromarray(depth))
                self.canvas.create_image(640, 0, image=photo_depth, anchor=tk.NW)
                self.canvas.photo_depth = photo_depth

    def quit_app(self):
        self.stop_camera()
        self.is_running = False
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthCameraApp(root)
    root.mainloop()
