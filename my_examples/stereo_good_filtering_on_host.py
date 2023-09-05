# Works ok on oak d lite

import numpy as np
import cv2
import depthai as dai

# Define pipeline
pipeline = dai.Pipeline()

# Define sources 
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create StereoDepth node and set defaults
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(True)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 100
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)




# Create output nodes
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
xoutDepth.input.setBlocking(False)

xoutConfidence = pipeline.create(dai.node.XLinkOut)
xoutConfidence.setStreamName("confidence")
xoutConfidence.input.setBlocking(False)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
stereo.confidenceMap.link(xoutConfidence.input)



with dai.Device(pipeline) as device:
    # Get output queues
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    # confidenceQueue = device.getOutputQueue(name="confidence", maxSize=4, blocking=False)
    confidenceQueue = device.getOutputQueue(name="confidence", maxSize=4, blocking=False)
    
    # Enable Laser dot projector on OAK D pro: 
    device.setIrLaserDotProjectorBrightness(765)
    device.setIrFloodLightBrightness(1500)
    min_avg = -1
    max_avg = -1

    while True:
        # Confidence threshold
        threshold = 5

        # Get the depth frame
        depth = depthQueue.get()
        depthFrame = depth.getFrame()
        
        # Check if depth frame contains valid data
        sum = np.sum(depthFrame==0)
        if sum > 0.7*depthFrame.shape[0]*depthFrame.shape[1]:
            continue
        # Get the confidence frame
        confidence = confidenceQueue.get()
        confidenceFrame = confidence.getFrame()

        depthFrame[confidenceFrame < threshold] = 0        
        
        # Split depth frame into 100 subframes 10x10
        res_m = depthFrame.shape[0]
        res_m_div = int(res_m/10)
        res_n = depthFrame.shape[1]
        res_n_div = int(res_n/10)
        new_depth_frame = np.zeros((res_m,res_n))
        subframes_list = np.zeros((10,10,res_m_div,res_n_div))
        for m in range(10):
            for n in range(10):
                # If more than 70 of frame is 0, make it black (0)
                if np.sum(depthFrame[m*res_m_div:(m+1)*res_m_div, n*res_n_div:(n+1)*res_n_div] == 0) > 0.7*res_m_div*res_n_div:
                    subframe = np.zeros((res_m_div,res_n_div))
                else:
                    subframe = depthFrame[m*res_m_div:(m+1)*res_m_div, n*res_n_div:(n+1)*res_n_div]
                    median = np.median(subframe[subframe>0])
                    subframe[subframe==0] = median
                subframes_list[m,n,:,:] = subframe
                new_depth_frame[m*res_m_div:(m+1)*res_m_div, n*res_n_div:(n+1)*res_n_div] = subframe                
        
        
        depthFrame = new_depth_frame

        if min_avg == -1:        
            min_avg = np.percentile(depthFrame[depthFrame>0], 2)
        else:
            min_avg = min_avg*0.8 + np.percentile(depthFrame[depthFrame>0], 2)*0.2

        if max_avg == -1:
            max_avg = np.percentile(depthFrame[depthFrame>0], 98)
        else:
            max_avg = max_avg*0.8 + np.percentile(depthFrame[depthFrame>0], 98)*0.2

        min_depth = min_avg
        max_depth = max_avg


        # min_depth = np.percentile(depthFrame[depthFrame>0], 10)
        # max_depth = np.percentile(depthFrame[depthFrame>0], 90)

        # normalize depth frame on valid depth values       
        depthFrame[depthFrame >0] = np.interp(depthFrame[depthFrame >0], (min_depth, max_depth), (0, 255)).astype(np.uint8)
        # apply color map to depth frame
        depthFrame = depthFrame.astype(np.uint8)
        depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

        # Apply a median filter to the confidence frame
        #depthFrame = cv2.medianBlur(depthFrame, 5)
                
        # display the depth frame
        cv2.imshow("depth", depthFrame)
        # display the confidence frame
        cv2.imshow("confidence", confidenceFrame)
        if cv2.waitKey(1) == ord('q'):
            break