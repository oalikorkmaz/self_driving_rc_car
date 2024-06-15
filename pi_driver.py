from lanes import *
import cv2
import RPi.GPIO as GPIO
import pigpio
from picamera2 import Picamera2
from libcamera import Transform
import serial
import time

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
    sh = False
    throttle_input = 1

    try:
        while True:
            cam = picam2.capture_array()
            frame = cv2.resize(cam, (640, 480))
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_masked = masked_image(frame_hsv, False)
            frame_canny = canny(frame_masked, False)
            perspective_image = perspective_transform(frame_canny, show=sh)
            frame_roi = region_of_interest(perspective_image, sh)
            lines = detect_lines(frame_roi)
            averaged_line = average_slope_intercept(frame, lines)
            line_frame = display_lines(frame, averaged_line)
            steering = steering_angle(frame, averaged_line, show=False)
            ser.write(f"{int(steering)}\n".encode('utf-8'))
            print("Steering", int(steering))
            yolo_model = yolo_detect(frame, model)
            combo_frame = cv2.addWeighted(yolo_model, 0.8, line_frame, 1, 1)
            cv2.imshow('Result', combo_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Close Serial Communication.")
        ser.close()

if __name__ == "__main__":
    main()
