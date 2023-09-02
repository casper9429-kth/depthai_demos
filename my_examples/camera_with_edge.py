# Use API 
# 1. Goal is to start and display cameras through API and send it to edge detection and display it

import cv2
import numpy as np
import depthai as dai

# 1. Start defining a pipeline 
# * camera souce, use rgb camera
# * use XLINK to send data to edge detection
# * use XLINK to send data to display
# * use edge detection
# * use XLINK to send edges to display
def onboard_camera_rgb():
    pipeline = dai.Pipeline()

    # Create Color Camera and set properties
    rgbCam = pipeline.create(dai.node.ColorCamera)
    rgbCam.setPreviewSize(300, 300)
    rgbCam.setInterleaved(False)
    rgbCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    rgbCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgbCam.setBoardSocket(dai.CameraBoardSocket.CAM_A)



    # Setup preview and link it to camera preview
    xout_cam_preview = pipeline.create(dai.node.XLinkOut)
    xout_cam_preview.setStreamName("cam_preview")
    xout_cam_preview.input.setBlocking(False)
    rgbCam.preview.link(xout_cam_preview.input)


    # Setup edge detection and link it to camera 
    edgeDetection = pipeline.create(dai.node.EdgeDetector)
    edgeDetection.setMaxOutputFrameSize(rgbCam.getVideoWidth() * rgbCam.getVideoHeight())
    # Create XLINK output for edge detection
    xout_edge = pipeline.create(dai.node.XLinkOut)
    xout_edge.setStreamName("edge_out")
    xout_edge.input.setBlocking(False)
    # Link camera to edge detection
    rgbCam.video.link(edgeDetection.inputImage)
    # Link edge detection to edge XLINK output
    edgeDetection.outputImage.link(xout_edge.input)


    # connect to device and start pipeline
    with dai.Device(pipeline) as device:
        
        # Get edge detection XLINK output stream 
        edgeQueue = device.getOutputQueue(name="edge_out", maxSize=8, blocking=False)
        # Get camera preview XLINK output stream
        camPreviewQueue = device.getOutputQueue(name="cam_preview", maxSize=8, blocking=False)
        while True:
            frame_edge = edgeQueue.get()
            frame_cam = camPreviewQueue.get()
            # Display edge detection
            cv2.imshow("edge", frame_edge.getCvFrame())
            # Display camera preview
            cv2.imshow("preview", frame_cam.getCvFrame())
            if cv2.waitKey(1) == ord('q'):
                break
        




def onboard_camera_mono():
    # Create pipeline
    pipeline = dai.Pipeline()
    # Create mono camera and set properties
    monoCam = pipeline.create(dai.node.MonoCamera)
    monoCam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoCam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    
    # Create XLINK output for mono camera to preview
    xout_mono = pipeline.create(dai.node.XLinkOut)
    xout_mono.setStreamName("mono")
    xout_mono.input.setBlocking(False)
    monoCam.out.link(xout_mono.input)
    
    
    # Create Edge Detection and set properties
    edgeDetection = pipeline.create(dai.node.EdgeDetector)
    edgeDetection.setMaxOutputFrameSize(monoCam.getResolutionWidth() * monoCam.getResolutionHeight()) # Important to make sure the output queue is large enough to support the full frame
    # link mono camera to edge detection
    monoCam.out.link(edgeDetection.inputImage)
    
    # Create XLINK output for edge detection
    xout_edge = pipeline.create(dai.node.XLinkOut)
    xout_edge.setStreamName("edge")
    xout_edge.input.setBlocking(False)
    # Link edge detection to edge XLINK output
    edgeDetection.outputImage.link(xout_edge.input)
    
    
    
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Get edge detection XLINK output stream
        out_edge = device.getOutputQueue(name="edge", maxSize=8, blocking=False)
        # Get mono camera XLINK output stream
        out_mono = device.getOutputQueue(name="mono", maxSize=8, blocking=False)
        while True:
            frame_edge = out_edge.get()
            frame_mono = out_mono.get()
            # Display edge detection
            cv2.imshow("edge", frame_edge.getCvFrame())
            # Display mono camera
            cv2.imshow("mono", frame_mono.getCvFrame())
            if cv2.waitKey(1) == ord('q'):
                break
            
def edge_detector_laptop_camera():
    """
    Use accelerated edge detection on laptop camera stream
    """
    # Create pipeline
    pipeline = dai.Pipeline()
    
    
    # Create edge detection and set properties
    edgeDetection = pipeline.create(dai.node.EdgeDetector)
    edgeDetection.setMaxOutputFrameSize(480*640)
    edgeDetection.setWaitForConfigInput(False)
    
    
    # Create xout for edge detection
    xout_edge = pipeline.create(dai.node.XLinkOut)
    xout_edge.setStreamName("edge_out")
    xout_edge.input.setBlocking(False)
    # Link edge detection to edge XLINK output
    edgeDetection.outputImage.link(xout_edge.input)
    
    
    # Create xin for edge detection
    xin_edge = pipeline.create(dai.node.XLinkIn)
    xin_edge.setStreamName("edge_config")
    xin_edge.out.link(edgeDetection.inputImage)
    
    # Get webcam video stream 
    cap = cv2.VideoCapture(0)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Get edge detection XLINK output stream
        out_edge = device.getOutputQueue(name="edge_out", maxSize=8, blocking=False)
        in_data = device.getInputQueue(name="edge_config", maxSize=8, blocking=False)
        while True:
            # Get frame from webcam
            ret, frame = cap.read()
            # Make frame grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Make frame uint8
            frame = frame.astype(np.uint8)
            # Convert frame to dai 
            dai_frame = dai.ImgFrame()
            dai_frame.setData(frame)
            dai_frame.setWidth(frame.shape[1])
            dai_frame.setHeight(frame.shape[0])
            
            dai_frame.setType(dai.RawImgFrame.Type.GRAY8)
            
            # Send frame to edge detection
            
            
            in_data.send(dai_frame)
            
            if out_edge.has():
                # Get edge detection frame
                frame_edge = out_edge.get()
                # Display edge detection
                cv2.imshow("edge", frame_edge.getCvFrame())
                if cv2.waitKey(1) == ord('q'):
                    break


    # # Show webcam video stream
    # while True:
    #     ret, frame = cap.read()
    #     cv2.imshow("preview", frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break    
    
onboard_camera_rgb()
#edge_detector_laptop_camera()