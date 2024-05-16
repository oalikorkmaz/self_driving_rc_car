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

ser = serial.Serial('/dev/ttyACM0', 9600)


def send_data(steering, throtle):
    

def main():
    # hata ayıklama için değer
    sh = False
    while True:
        cam = picam2.capture_array()

        frame = cv2.resize(cam, (640, 480))
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #maskeleme
        frame_masked = masked_image(frame_hsv, sh)

        #Canny Filtresi
        frame_canny = canny(frame_masked, sh)

        #İlgi alanını belirleme (ROI)
        frame_cropped = region_of_interest(frame_canny, sh)

        #Şeritleri algılama
        lines = detect_lines(frame_cropped)
    
        #Şeritlerin sol ve sağ da gruplanması
        averaged_line = average_slope_intercept(frame, lines)
        line_frame = display_lines(frame, averaged_line)
        # Açının oluşturulması
        steering_input_raw = steering_angle(frame, averaged_line, show=sh)
        print("Steering Input Raw", int(steering_input_raw))
        ser.write(str(int(steering_input_raw)).encode() + b'\n')
        
        combo_frame = cv2.addWeighted(frame, 0.8, line_frame, 1, 1)
        cv2.imshow('Result', combo_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


