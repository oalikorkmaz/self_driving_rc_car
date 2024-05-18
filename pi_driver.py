from lanes import *
import cv2
import RPi.GPIO as GPIO
import pigpio
from picamera2 import Picamera2
import serial
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (4608, 2592)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# son 'n' çıktıyı tutmak için liste. Fikir, hepsinin ortalamasını almak ve ardından nihai bir çıktı üretmektir,
# Bu, servo ve hızdaki ani değişiklikleri ve sarsıntıları yumuşatmaya yardımcı olur
steering_last_five = []
throttle_last_three = []

#Direksiyonu kalibre etmek için
steering_factor = 1

ser = serial.Serial('/dev/ttyUSB0', 9600)


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
            th = th - 25
            throttle_input = th
            print("Throttle Down: ", throttle_input)
        elif len(averaged_line) == 2:
            th = th + 25
            throttle_input = th
            print("Throttle Up: ", throttle_input)
        else:
            throttle_input = 0
            print("Throttle Stop: ", throttle_input)

        throttle_input = max(0, min(th, 125))  # Gaz değerini 0 ile 50 arasında sınırla
        steering_input_raw = max(0, min(steering_input_raw, 180))  # Açıyı 0 ile 180 arasında sınırla

        combo_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
        cv2.imshow('Result', combo_frame)
        ser.write(f"{int(steering_input_raw)},{int(throttle_input)}\n".encode())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


