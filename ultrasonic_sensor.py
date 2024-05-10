import RPi.GPIO as GPIO
import time

# TRIG_RIGHT = 10
# ECHO_RIGHT = 25
# TRIG_MID = 9
# ECHO_MID = 8
# TRIG_LEFT = 11
# ECHO_LEFT = 7

# def setup():
#     GPIO.setmode(GPIO.BCM)
#     GPIO.setup(TRIG_RIGHT, GPIO.OUT)
#     GPIO.setup(ECHO_RIGHT, GPIO.IN)
#     GPIO.setup(TRIG_MID, GPIO.OUT)
#     GPIO.setup(ECHO_MID, GPIO.IN)
#     GPIO.setup(TRIG_LEFT, GPIO.OUT)
#     GPIO.setup(ECHO_LEFT, GPIO.IN)
    

def measure_distance(TRIG, ECHO):
    
    GPIO.output(TRIG, GPIO.LOW)
    time.sleep(0.5)

    GPIO.output(TRIG, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG, GPIO.LOW)

    while GPIO.input(ECHO) == GPIO.LOW:
        pulse_start = time.time()

    while GPIO.input(ECHO) == GPIO.HIGH:
        pulse_end = time.time()

    # Mesafe hesapla (cm cinsinden)
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    
    return distance

def cleanup():
    GPIO.cleanup()

# if __name__ == '__main__':
#     try:
#         setup()
#         while True:
#             dist_right = measure_distance(TRIG_RIGHT, ECHO_RIGHT)
#             dist_mid = measure_distance(TRIG_MID, ECHO_MID)
#             dist_left = measure_distance(TRIG_LEFT, ECHO_LEFT)
#             print("Mesafe sag:", dist_right, "cm")
#             print("Mesafe orta:", dist_mid, "cm")
#             print("Mesafe sol:", dist_left, "cm")
#             time.sleep(1)
#     except KeyboardInterrupt:
#         cleanup()
