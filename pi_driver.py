from lanes import *
import cv2
import RPi.GPIO as GPIO
import pigpio
from picamera2 import Picamera2
import serial
import time

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (4608, 2592)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Initialize serial communication with Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600)

# To hold the last 'n' outputs
steering_last_five = []
throttle_last_three = []

# Calibration factor for steering
steering_factor = 1

# Function to smooth the list of values
def smooth_values(values, window_size):
    if len(values) < window_size:
        return sum(values) / len(values)
    else:
        return sum(values[-window_size:]) / window_size

def main():
    sh = False
    th = 0
    while True:
        cam = picam2.capture_array()
        frame = cv2.resize(cam, (640, 480))
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        frame_masked = masked_image(frame_hsv, sh)
        frame_canny = canny(frame_masked, sh)
        frame_cropped = region_of_interest(frame_canny, sh)
        lines = detect_lines(frame_cropped)
        averaged_line = average_slope_intercept(frame, lines)
        line_frame = display_lines(frame, averaged_line)
        steering_input_raw = steering_angle(frame, averaged_line, show=sh)
        print("Steering Input Raw", int(steering_input_raw))

        if len(averaged_line) == 1:
            th = th - 10  # Decrease throttle slightly
            throttle_input = th
            print("Throttle Down: ", throttle_input)
        elif len(averaged_line) == 2:
            th = th + 10  # Increase throttle slightly
            throttle_input = th
            print("Throttle Up: ", throttle_input)
        else:
            throttle_input = 0
            print("Throttle Stop: ", throttle_input)

        throttle_input = max(0, min(th, 50))  # Limit throttle between 0 and 50
        steering_input_raw = max(0, min(steering_input_raw, 180))  # Limit angle between 0 and 180

        # Append the current values to the lists
        steering_last_five.append(steering_input_raw)
        throttle_last_three.append(throttle_input)

        # Calculate the smooth values
        steering_input_smooth = smooth_values(steering_last_five, 5)
        throttle_input_smooth = smooth_values(throttle_last_three, 3)

        combo_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
        cv2.imshow('Result', combo_frame)

        # Send smoothed values to Arduino
        ser.write(f"{int(steering_input_smooth)},{int(throttle_input_smooth)}\n".encode())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
