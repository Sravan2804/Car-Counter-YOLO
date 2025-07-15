from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import numpy as np
from sort import *

def apply_mask(img, mask):
    # Ensure mask is grayscale and uint8
    if len(mask.shape) == 3:
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Resize mask to match image if needed
    if mask.shape != img.shape[:2]:
        mask = cv.resize(mask, (img.shape[1], img.shape[0]))

    # If image has 3 channels, apply mask to each channel
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv.bitwise_and(img, img, mask=mask)
    else:
        # For single-channel images
        return cv.bitwise_and(img, mask)
    


# cap = cv.VideoCapture(0) 
cap = cv.VideoCapture(r"D:\Projects\OpenCV\Object Detection Project\Car-Counter-YOLO\Assets\cars.mp4") #path to the video file


model = YOLO("D:\Projects\OpenCV\Object Detection Project\Car-Counter-YOLO\YOLO-weights\yolov8l.pt") # Load the YOLO model

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv.imread(r"D:\Projects\OpenCV\Object Detection Project\Car-Counter-YOLO\Assets\cars.png") #path to the mask image




#TRACKING

tracker = Sort(max_age = 20, min_hits= 3, iou_threshold=0.3)  # Initialize SORT tracker

limits = [400, 297, 673, 297 ] #only for the cars video, you can change it according to your needs

totalCount = [] # Initialize total count of vehicles

while True:
    success, img = cap.read()  # Read each frame
    imgRegion = apply_mask(img, mask)  # Apply the mask to the image

    imgGraphic = cv.imread(r"D:\Projects\OpenCV\Object Detection Project\Car-Counter-YOLO\Assets\graphic3.png", cv.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphic, (0, 0))  # Overlay the graphic on the image

    results = model(imgRegion, stream = True)
    
    detections = np.empty((0, 5))  # Initialize an empty array for detections (taken from SORT)

    for r in results:

        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            #cvzone
            w, h = x2-x1, y2-y1
            

            # Confidence score
            conf = math.ceil(box.conf[0]*100)/100 
            
            # classname
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "bus" or currentClass == "motorbike" or \
                currentClass == "truck" and conf > 0.4:
                
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(35, y1)), scale=0.8)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=2, colorR=(255, 0, 255))
                CurrentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, CurrentArray))

    resultsTracker = tracker.update(detections)

    cv.line(img, (limits[0], limits[1]),(limits[2], limits[3]), (0, 0, 255), 3) 

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=4, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale = 1.5, thickness = 3, offset=10) 
        #for checking the id of the object whether it is being tracked correctly

        cx, cy = x1+w//2, y1+h//2
        cv.circle(img, (cx,cy), 5, (0, 0, 255), cv.FILLED) # Draw center point of the bounding box

        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
            if totalCount.count(id) == 0:
                totalCount.append(id)  # Add the ID to the count list
                cv.line(img, (limits[0], limits[1]),(limits[2], limits[3]), (0, 255, 0), 3) 

        # cvzone.putTextRect(img, f'Count:{len(totalCount)}', (50,50), thickness=3)        
        cv.putText(img, str(len(totalCount)), (160, 100), cv.FONT_HERSHEY_COMPLEX, 3, (131, 247, 245), 10)
        print(result)

    cv.imshow("image", img)  # Display the video frame
    # cv.imshow("Region of Interest", imgRegion)  # Display the masked region
    cv.waitKey(1)