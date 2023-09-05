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
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setCamera("right")

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
#depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
#depth.initialConfig.setBilateralFilterSigma(65535)
# Filter invalid points (points with depth equal to 0)

depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

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

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(765)
    device.setIrFloodLightBrightness(1500)
    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="depth", maxSize=4, blocking=False)


    # Rollig average color for depth map
    min_depth = 0
    max_depth = 12000
    while True:
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDisparity.getCvFrame()
        # Remove 10 points at all edges
        frame = frame[10:-10, 10:-10]

        # Noramlize in 5 to 95 range
        min_depth = np.percentile(frame[frame>0], 5)*0.1 + min_depth*0.9
        max_depth = np.percentile(frame[frame>0], 95)*0.1 + max_depth*0.9
        frame[frame < min_depth] = min_depth
        frame[frame > max_depth] = max_depth
        frame = np.interp(frame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frame)

        if cv2.waitKey(1) == ord('q'):
            break