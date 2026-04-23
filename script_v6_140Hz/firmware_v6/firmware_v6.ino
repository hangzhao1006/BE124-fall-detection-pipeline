/*
 * BE124 Ankle Exoskeleton V6 - Binary UDP
 * =========================================
 * Same sensors as V5, but sends binary floats instead of ASCII.
 * Faster: no snprintf overhead, smaller packets.
 *
 * Packet format (128 bytes):
 *   Bytes 0-3:    Header magic 0xBE124DAA (uint32)
 *   Bytes 4-7:    timestamp_sec (uint32, epoch seconds)
 *   Bytes 8-9:    timestamp_ms  (uint16, milliseconds 0-999)
 *   Bytes 10-11:  frame_count   (uint16, rolling counter)
 *   Bytes 12-47:  thigh [acc_x,y,z, gyro_x,y,z, mag_x,y,z] (9 floats = 36 bytes)
 *   Bytes 48-83:  shank [acc_x,y,z, gyro_x,y,z, mag_x,y,z] (9 floats = 36 bytes)
 *   Bytes 84-119: foot  [acc_x,y,z, gyro_x,y,z, mag_x,y,z] (9 floats = 36 bytes)
 *   Bytes 120-131: foot euler [x,y,z] (3 floats = 12 bytes)
 *   Total: 4+4+2+2+36+36+36+12 = 132 bytes
 *
 * Board: Adafruit Feather ESP32-S3 2MB PSRAM
 * Partition: No OTA (2MB APP / 2MB SPIFFS)
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <time.h>
#include <Wire.h>
#include <Adafruit_ICM20X.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_Sensor.h>

// ============================================================
// CONFIG
// ============================================================
const char* WIFI_SSID     = "吴笑阳's iPhone";
const char* WIFI_PASSWORD = "wxy020212";
const char* LAPTOP_IP     = "172.20.10.10";
const int   UDP_PORT      = 12345;

// ============================================================
// Binary packet
// ============================================================
#define MAGIC 0xBE124DAA

#pragma pack(push, 1)
struct IMUFrame {
  uint32_t magic;
  uint32_t epoch_sec;
  uint16_t epoch_ms;
  uint16_t frame_id;
  float thigh[9];   // acc_xyz, gyro_xyz, mag_xyz
  float shank[9];
  float foot[9];
  float foot_euler[3]; // heading, roll, pitch
};
#pragma pack(pop)

// ============================================================
// PCA9548A
// ============================================================
#define MUX_ADDR 0x70
#define CH_THIGH 5
#define CH_SHANK 4
#define CH_FOOT  7

void muxSelect(uint8_t ch) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
  delayMicroseconds(100);
}

// ============================================================
// Sensors
// ============================================================
Adafruit_ICM20948 icmThigh;
Adafruit_ICM20948 icmShank;
Adafruit_BNO055   bnoFoot(55, 0x28, &Wire);

bool thighOK = false;
bool shankOK = false;
bool footOK  = false;

// ============================================================
// WiFi + UDP + NTP
// ============================================================
WiFiUDP udp;
bool ntpSynced = false;
unsigned long ntpSyncMillis = 0;
time_t ntpSyncEpoch = 0;
long ntpSyncMicros = 0;
uint16_t frameCount = 0;

bool connectWiFi() {
  Serial.print("# WiFi connecting to: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - t0 > 15000) {
      Serial.println("\n# WiFi FAILED");
      return false;
    }
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("# WiFi connected! IP: ");
  Serial.println(WiFi.localIP());
  return true;
}

bool syncNTP() {
  Serial.println("# Syncing NTP...");
  configTime(0, 0, "pool.ntp.org", "time.nist.gov", "time.google.com");

  unsigned long t0 = millis();
  struct tm ti;
  while (!getLocalTime(&ti)) {
    if (millis() - t0 > 10000) {
      Serial.println("# NTP FAILED");
      return false;
    }
    delay(500);
  }

  struct timeval tv;
  gettimeofday(&tv, NULL);
  ntpSyncEpoch = tv.tv_sec;
  ntpSyncMicros = tv.tv_usec;
  ntpSyncMillis = millis();

  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &ti);
  Serial.print("# NTP synced: ");
  Serial.println(buf);
  return true;
}

void getTimeParts(uint32_t &sec, uint16_t &ms) {
  if (!ntpSynced) {
    unsigned long m = millis();
    sec = m / 1000;
    ms = m % 1000;
    return;
  }
  unsigned long elapsed = millis() - ntpSyncMillis;
  unsigned long totalUs = ntpSyncMicros + (elapsed % 1000) * 1000;
  sec = (uint32_t)(ntpSyncEpoch + (elapsed / 1000) + (totalUs / 1000000));
  ms = (uint16_t)((totalUs % 1000000) / 1000);
}

// ============================================================
// Setup
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n# ========================================");
  Serial.println("# BE124 V6 - Binary UDP");
  Serial.print("# Packet size: ");
  Serial.print(sizeof(IMUFrame));
  Serial.println(" bytes");
  Serial.println("# ========================================");

  Wire.begin();
  Wire.setClock(400000);
  delay(500);

  // Multiplexer
  Serial.println("# Checking PCA9548A...");
  Wire.beginTransmission(MUX_ADDR);
  if (Wire.endTransmission() == 0) {
    Serial.println("# PCA9548A OK");
  } else {
    Serial.println("# ERROR: PCA9548A not found!");
    while (1) delay(1000);
  }

  // THIGH
  Serial.println("# Init THIGH (Ch0)...");
  muxSelect(CH_THIGH);
  delay(100);
  if (icmThigh.begin_I2C()) {
    icmThigh.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
    icmThigh.setGyroRange(ICM20948_GYRO_RANGE_500_DPS);
    icmThigh.setAccelRateDivisor(10);
    icmThigh.setGyroRateDivisor(10);
    thighOK = true;
    Serial.println("#   THIGH OK");
  } else {
    Serial.println("#   THIGH FAIL");
  }

  // SHANK
  Serial.println("# Init SHANK (Ch1)...");
  muxSelect(CH_SHANK);
  delay(100);
  if (icmShank.begin_I2C()) {
    icmShank.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
    icmShank.setGyroRange(ICM20948_GYRO_RANGE_500_DPS);
    icmShank.setAccelRateDivisor(10);
    icmShank.setGyroRateDivisor(10);
    shankOK = true;
    Serial.println("#   SHANK OK");
  } else {
    Serial.println("#   SHANK FAIL");
  }

  // FOOT
  Serial.println("# Init FOOT (Ch2)...");
  muxSelect(CH_FOOT);
  delay(500);
  if (bnoFoot.begin()) {
    bnoFoot.setExtCrystalUse(true);
    footOK = true;
    Serial.println("#   FOOT OK");
  } else {
    Serial.println("#   FOOT FAIL");
  }

  Serial.print("# Sensors: ");
  Serial.print(thighOK ? "THIGH " : "");
  Serial.print(shankOK ? "SHANK " : "");
  Serial.println(footOK ? "FOOT" : "");

  if (!thighOK && !shankOK && !footOK) {
    Serial.println("# ERROR: No sensors! Halting.");
    while (1) delay(1000);
  }

  // WiFi + NTP
  if (connectWiFi()) {
    ntpSynced = syncNTP();
    if (!ntpSynced) Serial.println("# WARNING: NTP failed, using millis()");
  } else {
    Serial.println("# WARNING: No WiFi. Serial only.");
  }

  udp.begin(UDP_PORT);

  Serial.println("# ========================================");
  Serial.println("# READY - binary streaming");
  Serial.println("# ========================================");
}

// ============================================================
// Main Loop
// ============================================================
unsigned long lastWiFiCheck = 0;

void loop() {
  unsigned long now = millis();

  if (now - lastWiFiCheck > 5000) {
    lastWiFiCheck = now;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("# WiFi lost, reconnecting...");
      connectWiFi();
    }
  }

  // Build frame
  IMUFrame frame;
  frame.magic = MAGIC;
  getTimeParts(frame.epoch_sec, frame.epoch_ms);
  frame.frame_id = frameCount++;

  // Zero out arrays
  memset(frame.thigh, 0, sizeof(frame.thigh));
  memset(frame.shank, 0, sizeof(frame.shank));
  memset(frame.foot, 0, sizeof(frame.foot));
  memset(frame.foot_euler, 0, sizeof(frame.foot_euler));

  // THIGH
  if (thighOK) {
    muxSelect(CH_THIGH);
    sensors_event_t a, g, m, t;
    icmThigh.getEvent(&a, &g, &t, &m);
    frame.thigh[0]=a.acceleration.x; frame.thigh[1]=a.acceleration.y; frame.thigh[2]=a.acceleration.z;
    frame.thigh[3]=g.gyro.x; frame.thigh[4]=g.gyro.y; frame.thigh[5]=g.gyro.z;
    frame.thigh[6]=m.magnetic.x; frame.thigh[7]=m.magnetic.y; frame.thigh[8]=m.magnetic.z;
  }

  // SHANK
  if (shankOK) {
    muxSelect(CH_SHANK);
    sensors_event_t a, g, m, t;
    icmShank.getEvent(&a, &g, &t, &m);
    frame.shank[0]=a.acceleration.x; frame.shank[1]=a.acceleration.y; frame.shank[2]=a.acceleration.z;
    frame.shank[3]=g.gyro.x; frame.shank[4]=g.gyro.y; frame.shank[5]=g.gyro.z;
    frame.shank[6]=m.magnetic.x; frame.shank[7]=m.magnetic.y; frame.shank[8]=m.magnetic.z;
  }

  // FOOT
  if (footOK) {
    muxSelect(CH_FOOT);
    imu::Vector<3> ac = bnoFoot.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    imu::Vector<3> gy = bnoFoot.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    imu::Vector<3> mg = bnoFoot.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
    imu::Vector<3> eu = bnoFoot.getVector(Adafruit_BNO055::VECTOR_EULER);
    frame.foot[0]=ac.x(); frame.foot[1]=ac.y(); frame.foot[2]=ac.z();
    frame.foot[3]=gy.x()/57.2958f; frame.foot[4]=gy.y()/57.2958f; frame.foot[5]=gy.z()/57.2958f;
    frame.foot[6]=mg.x(); frame.foot[7]=mg.y(); frame.foot[8]=mg.z();
    frame.foot_euler[0]=eu.x(); frame.foot_euler[1]=eu.y(); frame.foot_euler[2]=eu.z();
  }

  // Send binary packet
  if (WiFi.status() == WL_CONNECTED) {
    udp.beginPacket(LAPTOP_IP, UDP_PORT);
    udp.write((const uint8_t*)&frame, sizeof(frame));
    udp.endPacket();
  }

  // Serial debug (every 100 frames to not slow down)
  if (frame.frame_id % 100 == 0) {
    Serial.print("# Frame ");
    Serial.print(frame.frame_id);
    Serial.print(" | T_acc_z=");
    Serial.print(frame.thigh[2], 2);
    Serial.print(" S_acc_z=");
    Serial.print(frame.shank[2], 2);
    Serial.print(" F_acc_z=");
    Serial.println(frame.foot[2], 2);
  }
}
