import cv2
import csv
import collections
import time
import numpy as np
from tracker import *

class VideoProcessing:
    def __init__(self):
        # Initialize Tracker
        self.tracker = EuclideanDistTracker()
        # Cars Passed Images
        self.cars_snapped = []

        self.confThreshold =0.2
        self.nmsThreshold= 0.2

        self.font_color = (100, 150, 255)
        self.font_size = 0.5
        self.font_thickness = 2

        self.middle_line_position = 225   
        self.up_line_position = self.middle_line_position - 15
        self.down_line_position = self.middle_line_position + 15

        self.classesFile = "coco.names"
        self.classNames = open(self.classesFile).read().strip().split('\n')

        self.required_class_index = [2, 3, 5, 7]

        self.detected_classNames = []

        self.modelConfiguration = 'yolov3-320.cfg'
        self.modelWeigheights = 'yolov3-320.weights'

        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeigheights)

        np.random.seed(40)
        self.colors = np.random.randint(0, 255, size=(len(self.classNames), 3), dtype='uint8')

        self.passed_count = [0,0,0,0]

        self.passed_list = []

    def get_snapshot(self,frame,x,y,w,h):
        img = frame.copy()
        crop_img = img[y:y+h, x:x+w]
        return crop_img

    # Function for finding the center of a rectangle
    def find_center(self, x, y, w, h):
        x1=int(w/2)
        y1=int(h/2)
        cx = x+x1
        cy=y+y1
        return cx, cy
    
    def count_vehicle(self,box_id, img):
        x, y, w, h, id, index = box_id

        # Find the center of the rectangle for detection
        center = self.find_center(x, y, w, h)
        ix, iy = center

        if((id not in self.passed_list) and (((iy - self.middle_line_position )**2)<= 50)):
            self.passed_list.append(id)
            snap_shot = self.get_snapshot(img,x,y,w,h)
            self.cars_snapped.append(snap_shot)
            self.passed_count[index] += 1

    # Function for finding the detected objects from the network output
    def postProcess(self,outputs,img):
        height, width = img.shape[:2]
        boxes = []
        classIds = []
        confidence_scores = []
        detection = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if classId in self.required_class_index:
                    if confidence > self.confThreshold:
                        # print(classId)
                        w,h = int(det[2]*width) , int(det[3]*height)
                        x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                        boxes.append([x,y,w,h])
                        classIds.append(classId)
                        confidence_scores.append(float(confidence))

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, self.confThreshold, self.nmsThreshold)
        # print(classIds)
        for i in indices.flatten():
            
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            if(y<400):
                continue
            # print(x,y,w,h)

            #color = [int(c) for c in self.colors[classIds[i]]]
            name = self.classNames[classIds[i]]
            self.detected_classNames.append(name)
            # Draw classname and confidence score 
            #cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            detection.append([x, y, w, h, self.required_class_index.index(classIds[i])])

        # Update the tracker for each object
        boxes_ids = self.tracker.update(detection)
        #print(np.shape(boxes_ids))
        for box_id in boxes_ids:
            self.count_vehicle(box_id, img)
    def realTime(self,filename):
        
        # Initialize the videocapture object
        cap = cv2.VideoCapture(filename)
        input_size = 320
        while True:
            success, img = cap.read()
            if not success:
                break
            #img = cv2.resize(img,(0,0),None,0.5,0.5)
            ih, iw, channels = img.shape
            self.middle_line_position = int(ih*3/4)
            self.up_line_position = self.middle_line_position - int(ih/15)
            self.down_line_position = self.middle_line_position + int(ih/15)
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

            # Set the input of the network
            self.net.setInput(blob)
            layersNames = self.net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in self.net.getUnconnectedOutLayers()]
            # Feed data to the networkl
            outputs = self.net.forward(outputNames)
        
            # Find the objects from the network output
            self.postProcess(outputs,img)


            if cv2.waitKey(1) == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

    def process(self,file_path):
        self.realTime(file_path)
        return {
                "cars":self.passed_count[0],
                "motorbike":self.passed_count[1],
                "bus":self.passed_count[2],
                "truck":self.passed_count[3],
                "vehicles_snapped":len(self.cars_snapped),
            }

