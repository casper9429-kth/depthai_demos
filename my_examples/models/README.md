# How to train a myriad blob for oak d

## 1. Prepare the dataset
Use roboflow or similar

## 2. Upload dataset to Ultralytics Hub

## 3. Train it there

## 4. Download the model as pt 

## 5. Use myriad blob compiler to convert to myriad blob [Yolo](https://tools.luxonis.com/)
Set a resolution which must be divisible by 32, e.g. 416x416 and as close as possible to the aspect ratio of the input images. 
Aspect ratio is the ratio of width to height. 

e.g. 
```
oak d pro
12 mp : 4056x3040
aspect ratio: 4056/3040=1.33
```

possible resolutions: 
```
416x320 : aspect ratio: 1.3
672x512 : aspect ratio: 1.33
```

Experiment with different amounts of shaves and different resolutions to find the best combination of speed and accuracy.

According to my experiments, the best resolution for oak d lite is 416x320 and 4 shaves together with yolov8s, giving 10-15 fps. 