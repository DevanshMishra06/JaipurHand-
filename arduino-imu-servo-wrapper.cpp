#include <Wire.h>
#include <Servo.h>

const int MPU_ADDR = 0x68; // Standard I2C address for Grove 9DOF (MPU9250/ICM20600)
Servo myServo;
int servoPos = 90; // Start centered

unsigned long previousMillis = 0;
const long interval = 10; // 10ms = 100Hz sampling rate

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // Wake up the IMU
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // Power management register
  Wire.write(0);    // Set to 0 to wake up
  Wire.endTransmission(true);

  myServo.attach(9);
  myServo.write(servoPos);
}

void loop() {
  unsigned long currentMillis = millis();

  // 1. Strict 100Hz Data Acquisition Timer
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B); // Starting register for Accel X
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 14, true); // Request 14 registers

    if(Wire.available() == 14) {
      int16_t ax = Wire.read()<<8 | Wire.read();
      int16_t ay = Wire.read()<<8 | Wire.read();
      int16_t az = Wire.read()<<8 | Wire.read();
      int16_t temp = Wire.read()<<8 | Wire.read(); // Skip temperature
      int16_t gx = Wire.read()<<8 | Wire.read();
      int16_t gy = Wire.read()<<8 | Wire.read();
      int16_t gz = Wire.read()<<8 | Wire.read();

      // Stream to Python
      Serial.print(ax); Serial.print(",");
      Serial.print(ay); Serial.print(",");
      Serial.print(az); Serial.print(",");
      Serial.print(gx); Serial.print(",");
      Serial.print(gy); Serial.print(",");
      Serial.println(gz);
    }
  }

  // 2. Listen for PyTorch Commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'L') {
      servoPos = max(0, servoPos - 20);
      myServo.write(servoPos);
    } else if (cmd == 'R') {
      servoPos = min(180, servoPos + 20);
      myServo.write(servoPos);
    }
  }
}