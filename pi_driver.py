from lanes import *
import cv2
import RPi.GPIO as GPIO
import pigpio



# motor sürücü pinleri
Ena = 26
In1 = 6
In2 = 5

# servo pin
servo = 12

#GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(Ena, GPIO.OUT)
GPIO.setup(In1, GPIO.OUT)
GPIO.setup(In2, GPIO.OUT)

#pwmA'nın hız kontrolü
pwmA = GPIO.PWM(Ena, 100)
pwmA.start(0)

# servo setup
pwm = pigpio.pi()
pwm.set_mode(servo, pigpio.OUTPUT)
pwm.set_PWM_frequency(servo, 50)

# direksiyon için aralık belirleme
axis_start = 50
axis_end = -50
axis_range = axis_end - axis_start

# Gaz için aralık belirleme
throttle_start = 0
throttle_end = 100
throttle_range = throttle_end - throttle_start


# Araba servosu için aralık belirleme
steering_start = 900
steering_end = 2100
steering_range = steering_end - steering_start

# son 'n' çıktıyı tutmak için liste. Fikir, hepsinin ortalamasını almak ve ardından nihai bir çıktı üretmektir,
# Bu, servo ve hızdaki ani değişiklikleri ve sarsıntıları yumuşatmaya yardımcı olur
steering_last_five = []
throttle_last_three = []

#Direksiyonu kalibre etmek için
steering_factor = 1

def main():
    th = 0
    cam = cv2.VideoCapture("utils/video/test_video.mp4")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hata ayıklama için değer
        sh = False

        #maskeleme
        frame_masked = masked_image(frame_hsv, sh)

        #Canny Filtresi
        frame_canny = canny(frame_masked, sh)

        #İlgi alanını belirleme (ROI)
        frame_cropped = region_of_interest(canny_frame, sh)

        #Şeritleri algılama
        lines = detect_lines(cropped_frame)

        #Şeritlerin sol ve sağ da gruplanması
        averaged_line = average_slope_intercept(frame, lines)

        steering_input_raw = steering_angle(frame, averaged_line, show=sh)
        print("Steering Input Raw", int(steering_input_raw))

        #Kalibre etme
        current_axis = steering_input_raw * steering_factor
        print("Current Axis", int(current_axis))

        #direksiyon çıktısını servo açısına eşitleme (haritalandırma)
        steering_input = steering_start + (steering_range / axis_range) * (current_axis - axis_start)
        print("Steering Input", int(steering_input))

        st = int(steering_input)

        # son 5 değerle ortalama almak için listeye ekleme
        steering_last_five.append(st)
        if len(steering_last_five) > 5:
            steering_last_five.pop(0)

        st = int(sum(steering_last_five) / len(steering_last_five))

        # sadece bir şerit çizgisi tespit edilirse, aracın sert bir şekilde sağa veya sola gitmesi gerektiği anlamına gelir    
        # ayrıca steering_angle() sadece bir şerit varsa tek şeridin eğimini döndürür.
        # bu eğim aralığı iki şerit olduğunda farklıdır.
        # Bu yüzden burada 70 ila 20 arasında yeni bir aralık tanımlıyoruz ve servo açısını buna göre eşleştiriyoruz 
        if(len(averaged_line) == 1):
            print("One Lane Slope: ", steering_input_raw)
            axs = 70
            axe = 20
            axr = axe - axs
            st = steering_start + (steering_range / axr) * (steering_input_raw - axs)
            print("One Lane Steer: ", st)

            # Keskin bir dönüş için aracın yavaşlaması gerek
            th = th - 2
            throttle_input = th
            print("Throttle Down: ", throttle_input)
        else:
            # Eğer tek çizgi gözükmüyorsa yol açık demektir gaz arttırılabilir.
            th = th + 3
            throttle_input = th
            print("Throttle Up: ", throttle_input)
        
        # servonun menzil içerisinde kaldığından emin olmak için gerekli blok
        if st < 900:   
            st = 900
        if st > 2100:
            st = 2100

        # Nihai değerlerin atanması
        pwm.set_servo_pulsewidth(servo, st)
        pwmA.ChangeDutyCycle(int(throttle_input))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pwmA.ChangeDutyCycle(int(0))
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


