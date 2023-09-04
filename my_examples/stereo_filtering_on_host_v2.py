import numpy as np
import cv2
import depthai as dai

# Define pipeline
pipeline = dai.Pipeline()

# Define sources 
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create StereoDepth node and set defaults
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
stereo.setConfidenceThreshold(245)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(True)

# Create SpatialLocationCalculator node
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.inputDepth.setBlocking(False) 


# Create output nodes
## depth
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
xoutDepth.input.setBlocking(False)
## confidence
xoutConfidence = pipeline.create(dai.node.XLinkOut)
xoutConfidence.setStreamName("confidence")
xoutConfidence.input.setBlocking(False)
## spatial data
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xoutSpatialData.setStreamName("spatialData")
xoutSpatialData.input.setBlocking(False)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
stereo.confidenceMap.link(xoutConfidence.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.out.link(xoutSpatialData.input)

# Create Roi's for spatial location calculator
## Create 10x10 grid of roi's for spatial location calculator
N = 20
M = 20
for n in range(N):
    for m in range(M):
        config = dai.SpatialLocationCalculatorConfigData()
                
        lowerLeft = dai.Point2f((n)/N, (m)/M)
        upperRight = dai.Point2f((n+1)/N, (m+1)/M)
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 300
        config.depthThresholds.upperThreshold = 12000
        config.roi = dai.Rect(lowerLeft, upperRight)
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        
        # Confidence threshold for median calculation, valid only for median algorithm
        spatialLocationCalculator.initialConfig.addROI(config)


with dai.Device(pipeline) as device:
    # Get output queues
    ## depth queue
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    ## confidence queue
    confidenceQueue = device.getOutputQueue(name="confidence", maxSize=4, blocking=False)
    ## spatial data queue
    spatialDataQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    
    
    min_avg = -1
    max_avg = -1
    while True:
        # Confidence threshold
        # threshold = 5

        # Get the depth frame
        depth = depthQueue.get()
        depthFrame = depth.getFrame()
        
        # Get the confidence frame
        confidence = confidenceQueue.get()
        confidenceFrame = confidence.getFrame()

        # Get the spatial data
        spatialData = spatialDataQueue.get()
        spatialDataMap = spatialData.getSpatialLocations()

        # Create depth frame for visualization from only spatial data
        warningFrame = np.zeros((depthFrame.shape[0], depthFrame.shape[1],3), dtype=np.uint8)
        spatialDataDepthFrame = np.zeros((depthFrame.shape[0], depthFrame.shape[1]), dtype=np.uint8)
        for depthData in spatialDataMap:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrame.shape[1], height=depthFrame.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)
            coordinates = depthData.spatialCoordinates
            if np.sqrt(coordinates.x**2 + coordinates.y**2 + coordinates.z**2) < 800:
                warningFrame[ymin:ymax, xmin:xmax] = (0,0,255)            
            
            spatialDataDepthFrame[ymin:ymax, xmin:xmax] = coordinates.z/100#depthFrame[ymin:ymax, xmin:xmax]


        # Normalize depth frame for visualization
        min_depth = np.percentile(spatialDataDepthFrame[spatialDataDepthFrame>0], 1)
        max_depth = np.percentile(spatialDataDepthFrame[spatialDataDepthFrame>0], 99)
        spatialDataDepthFrame = np.interp(spatialDataDepthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        #spatialDataDepthFrame = np.interp(spatialDataDepthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)

        
        cv2.imshow("depthFrame", depthFrame)
        cv2.imshow("spatialDataDepthFrame", spatialDataDepthFrame)
        cv2.imshow("warningFrame", warningFrame)
        if cv2.waitKey(1) == ord('q'):
            break