#include <Arduino.h>
#include <HCSR04.h>
#include <Servo.h>

#define IN1 4
#define IN2 5
#define ENA 6
#define servoPin 11

UltraSonicDistanceSensor distanceSensorRight(2, 3); // trig, echo
UltraSonicDistanceSensor distanceSensorMid(7, 8); // trig, echo
UltraSonicDistanceSensor distanceSensorLeft(9,10); // trig, echo

Servo frontServo;

int distanceRight = 0;
int distanceMid = 0;
int distanceLeft = 0;
int steeringInt = 90;
int mapSteeringInt = 0;
int throttleInt = 0;
String textStr = "";
String receivedData = "";

void setup () {
    Serial.begin(115200);
    Serial.setTimeout(10);

    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);
    frontServo.attach(servoPin);
}

void motor_control(int throotle){
  if (throotle > 0){
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, throotle);
  }
  else if(throotle < 0){
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, -throotle);
  }
  else{
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);
  }
}

void measure_data(int distanceLeft, int distanceMid, int distanceRight){
  if ((distanceLeft >= 0 && distanceLeft <= 30) || (distanceMid >= 0 && distanceMid <= 30) || (distanceRight >= 0 && distanceRight <= 30)){
    //Serial.println("MOTOR DURDUR");
    motor_control(0);
  }
  else if(distanceLeft == -1 && distanceMid == -1 && distanceRight == -1){
    motor_control(0);
  }
  else{
    //Serial.println("motor calisiyor.");
    motor_control(68);
  }
}

void process_received_data(String data) {
  int commaIndex = data.indexOf(',');
  if (commaIndex > 0) {
    String steeringStr = data.substring(0, commaIndex);
    textStr = data.substring(commaIndex + 1);
    steeringInt = steeringStr.toInt();
    textStr.trim(); // Boşlukları kaldır
    Serial.println("Steering: " + String(steeringInt) + ", Text: " + textStr);
  }
}

void read_data() {
  while (Serial.available() > 0) {
    char incomingByte = Serial.read();
    if (incomingByte == '\n') { // Yeni satır karakteri ile veriyi sonlandır
      process_received_data(receivedData);
      receivedData = ""; // Veriyi sıfırla
    } else {
      receivedData += incomingByte; // Gelen byte'ı stringe ekle
    }
  }
}


void servo_control(){
  steeringInt = constrain(steeringInt, 40, 140);
  mapSteeringInt = map(steeringInt, 0, 180, 900, 2100);
  //Serial.println("Servo angle: " + String(steeringInt));
  //Serial.println("Map Servo angle: " + String(mapSteeringInt));
  frontServo.writeMicroseconds(mapSteeringInt);
}

void detection(){
  if (textStr == "stop") {
    motor_control(0);
    delay(1000);
    motor_control(68);
    Serial.println("stop");
  } else if (textStr == "orange") {
    motor_control(55);
    delay(1000);
    Serial.println("orange");
  } else if (textStr == "red") {
    motor_control(0);
    Serial.println("red");
    delay(1000);
  } else if (textStr == "green") {
    motor_control(68);
    delay(1000);
    Serial.println("green");
  }
}

void loop () {
    read_data();

    distanceLeft = distanceSensorLeft.measureDistanceCm();
    distanceMid = distanceSensorMid.measureDistanceCm();
    distanceRight = distanceSensorRight.measureDistanceCm();
    
    // Serial.println("Mesafe Sol: " + String(distanceLeft));
    // Serial.println("Mesafe Orta: " + String(distanceMid));
    // Serial.println("Mesafe Sag: " + String(distanceRight));
    
    measure_data(distanceLeft, distanceMid, distanceRight);
    delay(500);
    detection();
    servo_control();

}


