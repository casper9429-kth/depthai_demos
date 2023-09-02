import numpy as np
import cv2 as cv
import depthai as dai


# Create pipeline
pipeline = dai.Pipeline()

# Define souces, MonoCamera and RGBCamera
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
rgbCam = pipeline.create(dai.node.ColorCamera)

# Define stereo Node
stereo = pipeline.create(dai.node.StereoDepth)

# Define outputs: stereo
xout_stereo = pipeline.create(dai.node.XLinkOut)
xout_stereo.setStreamName("disparity")
xout_stereo.input.setBlocking(False)


# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

rgbCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgbCam.setBoardSocket(dai.CameraBoardSocket.RGB)
rgbCam.setInterleaved(False)
rgbCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.MedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.initialConfig.setExtendedDisparity(False)
stereo.initialConfig.setSubpixel(False)
stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
stereo.setLeftRightCheck(False)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(xout_stereo.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Create a buffer
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    while True:
        # Get the next frame
        inDisp = q.get()
        # Normalize in min-max range
        # Fill all -1 (invalid) pixels with surrounding valid pixels
        frame = inDisp.getFrame()
        minx = np.min(frame[frame != 0])
        maxx = np.max(frame[frame != 0])
        frame = np.interp(frame, (minx, maxx), (0, 255))


        # Get positive min and max values
        #frame = cv.normalize(inDisp.getFrame(), None, 0, 255, cv.NORM_MINMAX)
        # Convert to uint8
        frame = frame.astype(np.uint8)
        
        # Display the frame
        cv.imshow("disparity", frame)
        if cv.waitKey(1) == ord('q'):
            break


