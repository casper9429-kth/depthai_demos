import numpy as np
import cv2 
import depthai as dai

# StereoDepth and spatialLocationCalculator with multiple ROI
# This example shows usage of multiple ROI with spatialLocationCalculator and StereoDepth nodes

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and set resolutions and source
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create StereoDepth node and set defaults
depth = pipeline.create(dai.node.StereoDepth)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
#depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
#depth.initialConfig.setBilateralFilterSigma(65535)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(False)
depth.setSubpixel(True)
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

# Create spatialLocationCalculator node and set defaults
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.inputDepth.setBlocking(False)
# No config options for now, depth is already set by StereoDepth node
spatialLocationCalculator.setWaitForConfigInput(False)

# Create XLinkOut nodes
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xoutSpatialData.setStreamName("spatialData")

# Linking
## Link mono cameras to depth
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

## Link depth output to spatialLocationCalculator input
depth.depth.link(spatialLocationCalculator.inputDepth)

## Link spatialLocationCalculator output to XLinkOut
spatialLocationCalculator.out.link(xoutSpatialData.input)

## Might need to create a passthrough node to be able to connect to XLinkOut
depth.depth.link(xoutDepth.input)
#spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)

# Set up ROI's: NxM (N = horizontal, M = vertical)
N = 7
M = 7
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
        config.depthThresholds.lowerThreshold = 300
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(lowerLeft, upperRight)
        spatialLocationCalculator.initialConfig.addROI(config)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Get output queues
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qSpatialData = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)

    # Set Font
    text_color = (255, 255, 255)
    font_face = cv2.FONT_HERSHEY_TRIPLEX
    color_map = cv2.COLORMAP_RAINBOW

    while True:
        # Get depth data and perform normalization and color mapping
        inDepth = qDepth.get()
        depthFrame = inDepth.getFrame()
        min_depth = np.percentile(depthFrame[depthFrame>0], 1)
        max_depth = np.percentile(depthFrame[depthFrame>0], 99)
        depthFrame = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrame = cv2.applyColorMap(depthFrame, color_map)

        # Get spatial data
        inSpatialData = qSpatialData.get()
        spatialData = inSpatialData.getSpatialLocations()
        
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrame.shape[1], height=depthFrame.shape[0])
            xmin = int(roi.topLeft().x)
            xmax = int(roi.bottomRight().x)
            ymin = int(roi.topLeft().y)
            ymax = int(roi.bottomRight().y)
            coods = depthData.spatialCoordinates
            distance = np.sqrt(coods.x**2 + coods.y**2 + coods.z**2) 
            color_scale = np.interp(distance, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            color = cv2.applyColorMap(src=np.array([[color_scale]]),colormap= color_map)
            text_color = color[0,0,:].tolist()
            # Inverse color to make it readable on any background
            text_color = [255 - i for i in text_color]
            cv2.rectangle(depthFrame, (xmin, ymin), (xmax, ymax), text_color, thickness=2)
            cv2.putText(depthFrame, "{:.1f}m".format(distance/1000), (xmin + 10, ymin + 20), font_face, 0.6, text_color)
        cv2.imshow("depth", depthFrame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    




