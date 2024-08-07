#Import the neccesary libraries
import argparse
import cv2 
from smd.red import *
import os


# Construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run object detection network')
parser.add_argument("--prototxt", default = "MobileNetSSD.prototxt",
                                  help = 'Path to text network file')
parser.add_argument("--weights", default="MobileNetSSD.caffemodel",
                                 help='Path to weights')
parser.add_argument("--thr", default=0.2, type=float, help="Confidence threshold to filter out weak detections")
parser.add_argument("--close", default=0.35, type=float, help="How much robot is getting close. Lower values robot will stop further away")
parser.add_argument("--object", default='person', help="What object robot should follow. Objects are:"
                                                        "background, aeroplane, bicycle, bird, boat, bottle, bus, "
                                                        "car, cat, chair, cow, diningtable, dog, horse, motorbike, person, "
                                                        "pottedplant, sheep, sofa, train, tvmonitor")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Open camera device. 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height)
result = cv2.VideoWriter('recording.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 5, size)

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)


usb_port = "/dev/ttyUSB0"
# We need to read and write data to sub port jetson nano normally doesn't allow it
os.system('sudo chmod a+rw ' + usb_port)

class Robot:
    def __init__(self):
		# SMD setup
        self.port = usb_port
        self.m = Master(self.port)
        self.m.attach(Red(0))
        self.m.attach(Red(1))
        self.m.set_connected_modules(0,["Servo_1"])
        self.servo_pos = 90 #Current servo position
        self.last_error = 0 #Last error of motors used for PID
        self.las_dir = 0 #Last direction object is detected
        self.last_error_servo = 0 #Last error of servo used for PID

        # Motor setup
        # SMD 0 is left motor
        # SMD 1 is right motor
        self.m.set_shaft_cpr(0, 6533)
        self.m.set_shaft_rpm(0, 100)
        self.m.set_shaft_cpr(1, 6533)
        self.m.set_shaft_rpm(1, 100)
        self.m.set_operation_mode(0, OperationMode.Velocity)
        self.m.set_operation_mode(1, OperationMode.Velocity)
        self.m.enable_torque(0, True)
        self.m.enable_torque(1, True)
        self.m.set_servo(0, 1, 90)

        # Robot parameters
        self.rpm_left = 0 # Initial RPM for the left motor
        self.rpm_right = 0 # Intial RPM for the right motor
    
    # Drives the motors given error
    def motor_drive(self, error):
        kp = 0.2
        kd = 0.1
        # PID calculation for motors max 100 min -100
        self.rpm_left = min(100,max(-100,int(kp*error + 80 + kd*(self.last_error - error))))
        self.rpm_right = min(100,max(-100,int(-kp * error + 80 - kd*(self.last_error - error))))
        self.last_error = error

        # Motor speeds
        print("left speed:{},right speed:{}".format(self.rpm_left, self.rpm_right))
        self.m.set_velocity(1, self.rpm_left)
        self.m.set_velocity(0, -self.rpm_right)

    # Stops the motors
    def stop(self):
        self.m.set_velocity(1,0)
        self.m.set_velocity(0,0)
        
    #Calculates ratio between frame and the total are of object used for estimateing distance to object
    def calc_object_ratio_to_frame(self, xl, xr, yl, yr):
        return ((xr - xl) * (yr - yl))/(640*480)
    
    # AI object detection
    def detect(self):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_resized = cv2.resize(frame,(300,300)) # Resize frame for prediction
            
            # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
            # after executing this command our "blob" now has the shape: (1, 3, 300, 300)
            blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
            # Set to network the input blob 
            net.setInput(blob)
            # Prediction of network
            detections = net.forward()

            # Size of frame resize (300x300)
            cols = frame_resized.shape[1] 
            rows = frame_resized.shape[0]

            # For get the class and location of object detected, 
            # There is a fix index for class, location and confidence
            # value in @detections array .
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2] #Confidence of prediction 
                if confidence > args.thr: # Filter prediction 
                    class_id = int(detections[0, 0, i, 1]) # Class label

                    # Object location 
                    xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop   = int(detections[0, 0, i, 5] * cols)
                    yRightTop   = int(detections[0, 0, i, 6] * rows)
                    
                    # Factor for scale to original size of frame
                    heightFactor = frame.shape[0]/300.0  
                    widthFactor = frame.shape[1]/300.0 
                    
                    # Scale object detection to frame
                    xLeftBottom = int(widthFactor * xLeftBottom) 
                    yLeftBottom = int(heightFactor * yLeftBottom)
                    xRightTop   = int(widthFactor * xRightTop)
                    
                    # Draw rectangle and name of the object to the frame
                    yRightTop   = int(heightFactor * yRightTop)
                    if class_id in classNames:
                        label = classNames[class_id] + ": " + str(confidence)
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        yLeftBottom = max(yLeftBottom, labelSize[1])
                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))
                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                            (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                            (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                        
                        # Save the frame for the video
                        result.write(frame)
                        
                        # Check if object is what we are looking for
                    if classNames[class_id] == args.object:
                        return class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom
                    
            # If we don't find our object return 0's
            return 0,0,0,0,0,0
     
    # Look for human using the servo in the lates direction object is detected
    def look_for_object(self, dir):
        self.m.set_servo(1, 5, 120)
        if dir == 0:
            for i in reversed(range(0,10)):
                class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom = self.detect()
                if confidence != 0 and classNames[class_id] == args.object:
                    return i*10
                self.m.set_servo(0,1,i*10)
            while True:
                for i in range(0,19):
                    class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom = self.detect()
                    if confidence != 0 and classNames[class_id] == args.object:
                        return i*10
                    self.m.set_servo(0,1,i*10)
                for i in reversed(range(0,19)):
                    class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom = self.detect()
                    if confidence != 0 and classNames[class_id] == args.object:
                        return i*10
                    self.m.set_servo(0,1,i*10)
        else:
            for i in range(9,19):
                class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom = self.detect()
                if confidence != 0 and classNames[class_id] == args.object:
                    return i*10
                self.m.set_servo(0,1,i*10)
            while True:
                for i in reversed(range(0,19)):
                    class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom = self.detect()
                    if confidence != 0 and classNames[class_id] == args.object:
                        return i*10
                    self.m.set_servo(0,1,i*10)
                for i in range(0,19):
                    class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom = self.detect()
                    if confidence != 0 and classNames[class_id] == args.object:
                        return i*10
                    self.m.set_servo(0,1,i*10)
     
    # Turn the body of the robot to the specifed degree 
    def turn(self, deg):
        d = 0.5
        deg -= 90
        self.m.set_velocity(0, deg*d)
        self.m.set_velocity(1, deg*d)
        self.m.set_servo(0, 1, 95)
        self.servo_pos = 95
        time.sleep(1)
        self.m.set_velocity(0, 0)
        self.m.set_velocity(1, 0)
    
    # If a object is close track it with only the servo
    def track_object_close(self, error):
        kp = 0.02
        kd = 0.001
        
        # PID calculation for servo amx 180 min 0
        change = kp*error + kd*(self.last_error_servo - error)
        self.servo_pos = min(180, max(0, self.servo_pos + change))
        self.last_error_servo = error
        # Servo position
        print(f"Servo:{self.servo_pos}")
        self.m.set_servo(0, 1, int(self.servo_pos))
        
        # If object is getting out of frame turn the body
        if self.servo_pos > 95:
            self.las_dir = 1
        else:
            self.las_dir = 0
        if self.servo_pos > 160:
            self.turn(self.servo_pos)
        elif self.servo_pos < 20:
            self.turn(self.servo_pos)

    # Main loop of our object
    def run(self):
        while True:
            # Object detection
            class_id, confidence, yRightTop, yLeftBottom, xRightTop, xLeftBottom = self.detect()
            
            # If no object is detected look for it
            if confidence == 0:
                self.stop()
                deg = self.look_for_object(self.las_dir)
                self.turn(deg)
                continue
            
            # Calculates the error, error is difference between left and right conturs to the frame lenght
            rightDif = 640 - xRightTop
            leftDif = xLeftBottom
            error = rightDif - leftDif
            ratio = self.calc_object_ratio_to_frame(xLeftBottom,xRightTop,yLeftBottom,yRightTop)
            
            # Error is multiplied with ratio to make robot take faster turns at close and slower ay distance
            error *= ratio 
            
            # Determine if object is left or right from the error
            if error > 0:
                self.las_dir = 1
            else:
                self.las_dir = 0
            
            # If object is close enough stop and start tracking
            if ratio < args.close:
                self.m.set_servo(0, 1, 95)
                self.motor_drive(error)
            else:
                self.stop()
                self.track_object_close(error/ratio)

robo = Robot()
try:
    robo.run()
except KeyboardInterrupt:
    robo.stop()
    cap.release()
    result.release() 
