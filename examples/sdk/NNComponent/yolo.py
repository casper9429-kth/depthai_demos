from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('yolov8n_coco_640x352', color)
    oak.visualize([nn, color], scale=2 / 3, fps=True)  # 1080P -> 720P
    # oak.show_graph()
    oak.start(blocking=True)
