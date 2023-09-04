import numpy as np
import cv2
import depthai as dai
from DAIPipelineGraph import DAIPipelineGraph

pipeline_graph = DAIPipelineGraph( path="/home/casper/depthai_demos/my_examples/pipelines/basic_depth_and_confidence.json")

with dai.Device( pipeline_graph.pipeline ) as device:
    # Get output queues
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    # confidenceQueue = device.getOutputQueue(name="confidence", maxSize=4, blocking=False)
    confidenceQueue = device.getOutputQueue(name="confidence", maxSize=4, blocking=False)
    
    min_avg = -1
    max_avg = -1
    while True:
        # Confidence threshold
        threshold = 5

        # Get the depth frame
        depth = depthQueue.get()
        depthFrame = depth.getFrame()
        
        # Get the confidence frame
        confidence = confidenceQueue.get()
        confidenceFrame = confidence.getFrame()

        depthFrame[confidenceFrame < threshold] = 0        
        
        # Split depth frame into 100 subframes 10x10
        new_depth_frame = np.zeros((480,648))
        subframes_list = np.zeros((10,10,48,64))
        for m in range(10):
            for n in range(10):
                # If more than 70 of frame is 0, make it black (0)
                if np.sum(depthFrame[m*48:(m+1)*48, n*64:(n+1)*64] == 0) > 0.7*48*64:
                    subframe = np.zeros((48,64))
                else:
                    subframe = depthFrame[m*48:(m+1)*48, n*64:(n+1)*64]
                    median = np.median(subframe[subframe>0])
                    subframe[subframe==0] = median
                subframes_list[m,n,:,:] = subframe
                new_depth_frame[m*48:(m+1)*48, n*64:(n+1)*64] = subframe                
        
        
        depthFrame = new_depth_frame
        
        # get 95 and 5 percentile of depth frame
        if min_avg == -1:
            min_avg = np.percentile(depthFrame[depthFrame>0], 5)
        else:
            min_avg = min_avg*0.9 + np.percentile(depthFrame[depthFrame>0], 5)*0.1

        if max_avg == -1:
            max_avg = np.percentile(depthFrame[depthFrame>0], 95)
        else:
            max_avg = max_avg*0.9 + np.percentile(depthFrame[depthFrame>0], 95)*0.1

        min_depth = min_avg
        max_depth = max_avg

        # min_depth = np.percentile(depthFrame[depthFrame>0], 10)
        # max_depth = np.percentile(depthFrame[depthFrame>0], 90)

        # normalize depth frame on valid depth values       
        depthFrame[depthFrame >0] = np.interp(depthFrame[depthFrame >0], (min_depth, max_depth), (0, 255)).astype(np.uint8)
        # apply color map to depth frame
        depthFrame = depthFrame.astype(np.uint8)
        depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)
        
        # display the depth frame
        cv2.imshow("depth", depthFrame)
        # display the confidence frame
        cv2.imshow("confidence", confidenceFrame)
        if cv2.waitKey(1) == ord('q'):
            break