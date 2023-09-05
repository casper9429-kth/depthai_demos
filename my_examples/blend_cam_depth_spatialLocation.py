#!/usr/bin/env python3
# Works good on oak d pro with active ir projector
# Not enough for oak d lite

import cv2
import depthai as dai
import numpy as np

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = True
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
rgbCam = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xout = pipeline.create(dai.node.XLinkOut)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_spatial_1 = pipeline.create(dai.node.XLinkOut)


xout.setStreamName("depth")
xout_rgb.setStreamName("rgb")
xout_spatial_1.setStreamName("spatialData")
# Properties
rgbCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
rgbCam.setIspScale(1, 5)
rgbCam.setFps(20)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setCamera("left")
monoLeft.setFps(20)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")
monoRight.setFps(20)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration


# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
#depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
#depth.initialConfig.setBilateralFilterSigma(1000)
# Filter invalid points (points with depth equal to 0)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.setRectifyEdgeFillColor(0) # Black, to better see the cutout
depth.setDepthAlign(dai.CameraBoardSocket.RGB)
config = depth.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 100
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
depth.initialConfig.set(config)

# Create a spatial location calculator
spatialLocationCalculator_1 = pipeline.create(dai.node.SpatialLocationCalculator)
spatialLocationCalculator_1.setWaitForConfigInput(False)
spatialLocationCalculator_1.inputDepth.setBlocking(False)

# Config spatial calculator
N = 7 # 14
M = 7 # 14
safty_factor = 0.5
for n in range(N):
    for m in range(M):        
        # If n = 0, set left_offset to 0.1/N (10% of the image width), otherwise set it to 0
        left_offset = (safty_factor)/N if n == 0 else 0
        # If m = 0, set top_offset to 0.1/M (10% of the image height), otherwise set it to 0
        top_offset = (safty_factor)/M if m == 0 else 0
        # If n = N-1, set right_offset to 0.1/N (10% of the image width), otherwise set it to 0
        right_offset = (safty_factor)/N if n == N-1 else 0
        # If m = M-1, set bottom_offset to 0.1/M (10% of the image height), otherwise set it to 0
        bottom_offset = (safty_factor)/M if m == M-1 else 0
        
        # Set ROI
        lowerLeft = dai.Point2f(left_offset + (n)/N, top_offset + (m)/M)
        upperRight = dai.Point2f(1 - right_offset - (N-n-1)/N, 1 - bottom_offset - (M-m-1)/M)
                
        # lowerLeft = dai.Point2f((n)/N, (m)/M)
        # upperRight = dai.Point2f((n+1)/N, (m+1)/M)
        config = dai.SpatialLocationCalculatorConfigData()
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
        config.depthThresholds.lowerThreshold = 300
        config.depthThresholds.upperThreshold = 10000
        # set confidence threshold 
        config.roi = dai.Rect(lowerLeft, upperRight)
        spatialLocationCalculator_1.initialConfig.addROI(config)

                                                  

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(xout.input)
rgbCam.video.link(xout_rgb.input)
depth.depth.link(spatialLocationCalculator_1.inputDepth)
spatialLocationCalculator_1.out.link(xout_spatial_1.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
        if lensPosition:
            rgbCam.initialControl.setManualFocus(lensPosition)
    except:
        raise

    # Auto set IrLasers
    device.setIrLaserDotProjectorBrightness(765)
    #device.setIrFloodLightBrightness(1500)
    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="depth", maxSize=10, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=10, blocking=False)
    qSpatial_1 = device.getOutputQueue(name="spatialData", maxSize=10, blocking=False)
    # Rollig average color for depth map
    min_depth = 100
    max_depth = 12000
    while True:
        # Find sync frames
        
        frame_que = q.get()
        frame = frame_que.getCvFrame()
        rgb_que = qRgb.get()
        rgb = rgb_que.getCvFrame()

        # Noramlize in 5 to 95 range
        min_depth = np.percentile(frame[frame>0], 5)*0.1 + min_depth*0.9
        max_depth = np.percentile(frame[frame>0], 95)*0.1 + max_depth*0.9
        frame[frame < min_depth] = min_depth
        frame[frame > max_depth] = max_depth
        frame = np.interp(frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frame)
        cv2.imshow("rgb", rgb)
        # Blend rgb and depth
        frame = cv2.addWeighted(frame, 0.6, rgb, 0.4, 0)
        cv2.imshow("blend", frame)

        # Get spatial data

        spatial_1_que = qSpatial_1.get()
        spatial_1_data = spatial_1_que.getSpatialLocations()
        spatial_1_data = sorted(spatial_1_data, key=lambda depthData: depthData.spatialCoordinates.z)        
        for i,depthData in enumerate(spatial_1_data):
            
            
            roi = depthData.config.roi
            roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
            xmin = int(roi.topLeft().x)
            xmax = int(roi.bottomRight().x)
            ymin = int(roi.topLeft().y)
            ymax = int(roi.bottomRight().y)
            coods = depthData.spatialCoordinates
            distance = np.sqrt(coods.x**2 + coods.y**2 + coods.z**2) 
            color_scale = np.interp(distance, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            color = cv2.applyColorMap(src=np.array([[color_scale]]),colormap= cv2.COLORMAP_JET)
            text_color = color[0,0,:].tolist()
            # Inverse color to make it readable on any background
            text_color = [255 - i for i in text_color]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), text_color, thickness=2)
            cv2.putText(frame, "{:.1f}m".format(distance/1000), (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, text_color)
            if i == 0:
                cv2.rectangle(rgb, (xmin, ymin), (xmax, ymax), text_color, thickness=2)
                cv2.putText(rgb, "Closest object", (xmin + 10, ymin + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6, text_color)
                
        cv2.imshow("blend with grid", frame)

        cv2.imshow("rgb with grid", rgb)



        



        if cv2.waitKey(1) == ord('q'):
            break