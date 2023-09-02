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
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_rgb.input.setBlocking(False)

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

# Crop the rgb camera to fit the stereo output
rgbCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgbCam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
rgbCam.setInterleaved(False)
rgbCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
rgbCam.video.link(xout_rgb.input)


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
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        # Get the next frame
        inDisp = q.get()
        # Normalize in min-max range
        # Fill all -1 (invalid) pixels with surrounding valid pixels
        frame = inDisp.getFrame()
        frame = (frame * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        # Convert to uint8
        frame = frame.astype(np.uint8)        
        # Make it 3 channels JET
        frame = cv.applyColorMap(frame, cv.COLORMAP_JET)
        # Display the frame
        cv.imshow("disparity", frame)        
        # Show rgb
        frameRgb = qRgb.get().getCvFrame()
        cv.imshow("rgb", frameRgb)
        # Downsample frameRgb to match frame size
        frameRgb = cv.resize(frameRgb, (frame.shape[1], frame.shape[0]))

        # Blend frame and frameRgb
        frame = cv.addWeighted(frame, 0.5, frameRgb, 0.5, 0)
        cv.imshow("blend", frame)


        if cv.waitKey(1) == ord('q'):
            break


