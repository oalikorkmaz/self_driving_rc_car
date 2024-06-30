
import cv2
import RPi.GPIO as GPIO
import pigpio
from picamera2 import Picamera2
from libcamera import Transform
import serial
import time

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

from lanes import *

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1920, 1080)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Initialize serial communication with Arduino
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1.0)
time.sleep(3)
ser.reset_input_buffer()

def main():
    sh = True
    throttle_input = 1
    pos = (20, 60)
    font = cv2.FONT_HERSHEY_SIMPLEX

    height = 1.5
    weight = 3
    myColor = (255, 0, 0)

    fps = 0
    tStart = time.time()
    #model = 'sign.tflite'
    #num_threads = 4
    #text=""

    #base_options = core.BaseOptions(file_name=model, use_coral=False, num_threads=num_threads)
    #detection_options = processor.DetectionOptions(max_results=4, score_threshold=0.3)
    #options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    #detector = vision.ObjectDetector.create_from_options(options)

    try:
        while True:
            
            frame = picam2.capture_array()
            frame = cv2.resize(frame, (640, 480))
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            frame_masked = masked_image(frame_hsv, False)
            frame_canny = canny(frame_masked, False)
            #perspective_image = perspective_transform(frame_canny, show=sh)
            frame_roi = region_of_interest(frame_canny, sh)
            lines = detect_lines(frame_roi)
            averaged_line = average_slope_intercept(frame, lines)
            line_frame = display_lines(frame, averaged_line)
            steering = steering_angle(frame, averaged_line, show=False)
            cv2.putText(frame, str(int(fps)) + 'FPS', pos, font, height, myColor, weight)
            

            #frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frameTensor = vision.TensorImage.create_from_array(frameRGB)
            #detections = detector.detect(frameTensor)
            #image, result_texts = utils.visualize(frame, detections)
            
            
            #text = result_texts[0] if result_texts else "none"
            text="none"
            ser.write(f"{int(steering)}, {text}\n".encode('utf-8'))
            print("Steering", int(steering))
            heading_image = display_heading_line(line_frame, steering)
            cv2.imshow("heading line",heading_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            tEnd = time.time()
            loopTime = tEnd - tStart
            fps = 0.9 * fps + 0.1 * (1 / loopTime)
            tStart = time.time()

        picam2.stop()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Close Serial Communication.")
        ser.close()

if __name__ == "__main__":
    main()

