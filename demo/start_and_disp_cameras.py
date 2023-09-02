# Goal is to start and display cameras through API and SDKapi_hq
import depthai as dai
from depthai_sdk import OakCamera
import cv2
import depthai as dai
import numpy as np

# SDK or API
type = "sdk"



if type.lower() == "sdk":
    with OakCamera() as oak:
        color = oak.camera('color', resolution='1080p',fps=30)
        left = oak.camera('left', resolution='400p')
        right = oak.camera('right', resolution='400p')
        oak.visualize([color, left, right], fps=True)
        oak.start(blocking=True)
elif type.lower() == "api_preview":
    # Start defining a pipeline (in this case, a simple one with just a ColorCamera)
    pipeline = dai.Pipeline()
    # Define a source - color camera
    colorCam = pipeline.create(dai.node.ColorCamera)
    # Create a source - color camera
    colorCam.setPreviewSize(300, 300)
    colorCam.setInterleaved(False)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Create outputs
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    # set outut name
    xoutVideo.setStreamName("video")
    # Link plugins CAM -> XLINK 
    colorCam.preview.link(xoutVideo.input)    
    
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Get the queue for output
        qVideo = device.getOutputQueue(name="video", maxSize=8, blocking=False)
        
    
    
 
        while True:
            # Get the next frame
            inVideo = qVideo.get()
            # frame = inVideo.getFrame()
            # frame = np.einsum("cxy->xyc", frame)
            frame = inVideo.getCvFrame()
            cv2.imshow("video", frame)
            if cv2.waitKey(1) == ord('q'):
                break
elif type.lower() == "api_hq":
    pipeline = dai.Pipeline()
    # define a source - color camera
    colorCam = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    
    # set outut name
    xoutVideo.setStreamName("video")
    
    # Properties
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setInterleaved(False)
    colorCam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # Downscales frame to half size
    #colorCam.setVideoSize(1000, 1000)
    # scale down to 300x300
    #colorCam.setIspScale(1,5)
    
    # link camera to XLINK
    colorCam.video.link(xoutVideo.input)
    
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Get the queue for output
        qVideo = device.getOutputQueue(name="video", maxSize=8, blocking=False)
        
        while True:
            # Get the next frame
            inVideo = qVideo.get()
            # frame = inVideo.getFrame()
            # frame = np.einsum("cxy->xyc", frame)
            frame = inVideo.getCvFrame()
            cv2.imshow("video", frame)
            if cv2.waitKey(1) == ord('q'):
                break

