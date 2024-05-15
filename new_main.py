import cv2
import RPi.GPIO as GPIO
import pigpio
from picamera2 import Picamera2
import time

class UltrasonicMotorController:
    def __init__(self):
        self.TRIG_RIGHT = 10
        self.ECHO_RIGHT = 25
        self.TRIG_MID = 9
        self.ECHO_MID = 8
        self.TRIG_LEFT = 11
        self.ECHO_LEFT = 7
        self.servo = 12
        self.Ena = 26
        self.In1 = 6
        self.In2 = 5
        self.pwmA = GPIO.PWM(self.Ena, 100)
        self.pwmA.start(0)
        self.pwm = pigpio.pi()
        self.pwm.set_mode(self.servo, pigpio.OUTPUT)
        self.pwm.set_PWM_frequency(self.servo, 50)
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.Ena, GPIO.OUT)
        GPIO.setup(self.In1, GPIO.OUT)
        GPIO.setup(self.In2, GPIO.OUT)
        GPIO.setup(self.TRIG_RIGHT, GPIO.OUT)
        GPIO.setup(self.ECHO_RIGHT, GPIO.IN)
        GPIO.setup(self.TRIG_MID, GPIO.OUT)
        GPIO.setup(self.ECHO_MID, GPIO.IN)
        GPIO.setup(self.TRIG_LEFT, GPIO.OUT)
        GPIO.setup(self.ECHO_LEFT, GPIO.IN)

    def stop_motor(self):
        self.pwmA.ChangeDutyCycle(0)

    def start_motor(self, throttle_input):
        self.pwmA.ChangeDutyCycle(int(throttle_input))

    def control_steering(self, steering_input):
        if steering_input < 900:
            steering_input = 900
        elif steering_input > 2100:
            steering_input = 2100
        self.pwm.set_servo_pulsewidth(self.servo, steering_input)

    def check_sensors(self):
        distances = []
        for i in range(3):
            GPIO.output(self.TRIG_RIGHT, GPIO.LOW)
            time.sleep(0.00001)
            GPIO.output(self.TRIG_RIGHT, GPIO.HIGH)
            time.sleep(0.00001)
            GPIO.output(self.TRIG_RIGHT, GPIO.LOW)
            while GPIO.input(self.ECHO_RIGHT) == GPIO.LOW:
                start_time = time.time()
            while GPIO.input(self.ECHO_RIGHT) == GPIO.HIGH:
                end_time = time.time()
            distance = ((end_time - start_time) * 34300) / 2
            distances.append(distance)
        return distances

    def main_loop(self):
        while True:
            distances = self.check_sensors()
            if min(distances) <= 12:
                self.stop_motor()
            else:
                self.start_motor(throttle_input)  # throttle_input değişkenini uygun bir şekilde tanımlayın
            self.control_steering(steering_input)  # steering_input değişkenini uygun bir şekilde tanımlayın
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop_motor()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = UltrasonicMotorController()
    controller.main_loop()
