#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter

"""
This script shows how to:
* Use the neural network to detect objects on RGB camera and get spatial location coordinates: x,y,z relative to the center of depth map.
* Use the spatial location coordinates to find the closest object to the camera.
* Create a depth map from the stereo pair.
* Blend the RGB and depth frames.
"""

# Config path and model path: change to your own paths
#Get path to this file
curr_path = Path(__file__).resolve().parent

# configPath = '/home/casper/depthai_demos/my_examples/models/result_low_res/yolov8n_coco.json'
# nnPath = "/home/casper/depthai_demos/my_examples/models/result_low_res/yolov8n_coco_openvino_2022.1_6shave.blob"#args.model

#Using the curr_path variable, get the below paths dynamically
configPath = str(curr_path) + '/models/result_low_res/yolov8n_coco.json'
nnPath = str(curr_path) + '/models/result_low_res/yolov8n_coco_openvino_2022.1_6shave.blob'


# parse config
with Path(configPath).open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})
labelMap = labels

# Sync pictures and NN output
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs: 
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)
objectTracker = pipeline.create(dai.node.ObjectTracker)

# Create outputs
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutOT = pipeline.create(dai.node.XLinkOut)

trackerOut = pipeline.create(dai.node.XLinkOut)

# Set names for streams
xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")
trackerOut.setStreamName("tracklets")
xoutOT.setStreamName("preview_ot")

# Camera node Config
camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # Or THE_4_K (slower)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Stereo node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(W, H)
stereo.setSubpixel(True)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Align depth map to RGB camera
config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 100
config.postProcessing.thresholdFilter.maxRange = 15000
config.postProcessing.decimationFilter.decimationFactor = 2 # speedup
stereo.initialConfig.set(config)


# Spatial location calculator
## Spatial location calculator 
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(15000)
##  Yolo specific parameters
spatialDetectionNetwork.setConfidenceThreshold(confidenceThreshold)
spatialDetectionNetwork.setNumClasses(classes)
spatialDetectionNetwork.setCoordinateSize(coordinates)
spatialDetectionNetwork.setAnchors(anchors)
spatialDetectionNetwork.setAnchorMasks(anchorMasks)
spatialDetectionNetwork.setIouThreshold(iouThreshold)
spatialDetectionNetwork.setBlobPath(nnPath)
spatialDetectionNetwork.setNumInferenceThreads(2)
spatialDetectionNetwork.input.setBlocking(False)

## Object tracker speficic parameters
# https://docs.luxonis.com/projects/api/en/latest/components/nodes/object_tracker/ 
#Get the index of the person class in labels
ObjectOfInterest = "spoon"
IndexOfInterest = labelMap.index(ObjectOfInterest)
objectTracker.setDetectionLabelsToTrack([IndexOfInterest]) # Track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)


# Create spatialLocationCalculator node and set defaults
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.inputDepth.setBlocking(False)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xoutSpatialData.setStreamName("gridData")



# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

spatialLocationCalculator.out.link(xoutSpatialData.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

#Object tracker, OT
objectTracker.passthroughTrackerFrame.link(xoutOT.input)

spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame) #Want the frame to be the passtrough frame of the detection network

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
objectTracker.out.link(trackerOut.input)

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
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
        config.roi = dai.Rect(lowerLeft, upperRight)
        spatialLocationCalculator.initialConfig.addROI(config)


# Connect to device and start pipeline
with dai.Device(pipeline,usb2Mode=True) as device:
    device.setIrLaserDotProjectorBrightness(765)
    #device.setIrFloodLightBrightness(1500)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="gridData", maxSize=4, blocking=False)
    trackletsQueue = device.getOutputQueue(name="tracklets", maxSize=4, blocking=False)
    previewOTQueue = device.getOutputQueue(name="preview_ot", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True
    min_depth = 400
    max_depth = 15000
    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()
        gridCells = spatialCalcQueue.get().getSpatialLocations()
        inPreviewOT = previewOTQueue.get()
        inTracklets = trackletsQueue.get()
        

        if printOutputLayersOnce:
            toPrint = 'Output layer names:'
            for ten in inNN.getAllLayerNames():
                toPrint = f'{toPrint} {ten},'
            print(toPrint)
            printOutputLayersOnce = False



        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters
        depth_downscaled = depthFrame[::4] # Faster computation        
        if (depth_downscaled == 0).all():
            continue
        
        min_depth = 0.1*np.percentile(depth_downscaled[depth_downscaled != 0], 5) + 0.9*min_depth # Rolling average 
        max_depth = 0.1*np.percentile(depth_downscaled, 95) + 0.9*max_depth                       # Rolling average
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:

            # Uncomment to get spatial coordinate center of bounding box
            # roiData = detection.boundingBoxMapping
            # roi = roiData.roi
            # roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            # topLeft = roi.topLeft()
            # bottomRight = roi.bottomRight()
            # xmin = int(topLeft.x)
            # ymin = int(topLeft.y)
            # xmax = int(bottomRight.x)
            # ymax = int(bottomRight.y)
            # cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 4)

            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            label = labelMap[detection.label]
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            #cv2.rectangle(depthFrameColor, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)


        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        # cv2.imshow("depth", depthFrameColor)
        #cv2.imshow("rgb", frame)
        # Blend RGB and Depth
        frame_blend = cv2.addWeighted(frame, 0.7, depthFrameColor, 0.3, 0)
        cv2.imshow("rgb", frame_blend)


        # Sort grid cells by distance to camera
        gridCells.sort(key=lambda x: x.spatialCoordinates.z**2 + x.spatialCoordinates.x**2 + x.spatialCoordinates.y**2)
        # Show grid cells on depth frame
        for i,depthData in enumerate(gridCells):    
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            xmax = int(roi.bottomRight().x)
            ymin = int(roi.topLeft().y)
            ymax = int(roi.bottomRight().y)
            coods = depthData.spatialCoordinates
            distance = np.sqrt(coods.x**2 + coods.y**2 + coods.z**2) 
            color_scale = np.interp(distance, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            color1 = cv2.applyColorMap(src=np.array([[color_scale]]),colormap= cv2.COLORMAP_JET)
            text_color = color1[0,0,:].tolist()
            # Inverse color to make it readable on any background
            text_color = [255 - i for i in text_color]
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), text_color, thickness=2)
            cv2.putText(depthFrameColor, "{:.1f}m".format(distance/1000), (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color)
        cv2.imshow("depth_with_grid", depthFrameColor)

        # Show closest grid cell on frame
        closest = gridCells[0]
        roi = closest.config.roi
        roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
        xmin = int(roi.topLeft().x)
        xmax = int(roi.bottomRight().x)
        ymin = int(roi.topLeft().y)
        ymax = int(roi.bottomRight().y)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)
        cv2.putText(frame, "Closest grid cell", (xmin + 10, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color)
        cv2.imshow("rgb_with_grid", frame)

        # Object tracking window
        frameOT = inPreviewOT.getCvFrame()
        colorOT = (255,0,0)
        trackletsData = inTracklets.tracklets
        for t in trackletsData:
            roi = t.roi.denormalize(frameOT.shape[1], frameOT.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(frameOT, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frameOT, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frameOT, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frameOT, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            # cv2.putText(frameOT, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(frameOT, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            # cv2.putText(frameOT, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.putText(frameOT, "NN fps: {:.2f}".format(fps), (2, frameOT.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.putText(frameOT, "# tracklets: {:.2f}".format(len(trackletsData)), (frameOT.shape[1]-150, (frameOT.shape[0] - 4)), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.imshow("tracker", frameOT)

        if cv2.waitKey(1) == ord('q'):
            break