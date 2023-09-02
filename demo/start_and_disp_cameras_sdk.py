# Goal is to start and display cameras through API and SDK
import depthai as dai
from depthai_sdk import OakCamera


# SDK 
with OakCamera() as oak:
    color = oak.camera('color', resolution='1080p',fps=30)
    left = oak.camera('left', resolution='400p')
    right = oak.camera('right', resolution='400p')
    oak.visualize([color, left, right], fps=True)
    oak.start(blocking=True)
    
    

