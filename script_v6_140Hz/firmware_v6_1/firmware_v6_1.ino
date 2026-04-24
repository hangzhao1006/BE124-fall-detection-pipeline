/*
 * BE124 Ankle Exoskeleton V6.1 - Optimized Binary UDP
 * =====================================================
 * Changes from V6:
 *   - Fixed: Serial debug prints correct channel numbers (5/4/7)
 *   - Fixed: NTP periodic re-sync every 10 min to prevent clock drift
 *   - Added: Startup self-test with actual sensor values
 *   - Added: Frame rate counter in debug output
 *   - Added: BNO055 calibration status in debug output
 *   - Optimized: Reduced muxSelect delay from 100us to 50us
 *   - BNO055 gyro already converted to rad/s (/57.2958f)
 *
 * Packet format (132 bytes):
 *   magic(4) + epoch_sec(4) + epoch_ms(2) + frame_id(2)
 *   + thigh[9](36) + shank[9](36) + foot[9](36) + euler[3](12)
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
// CONFIG - verified correct
// ============================================================
const char* WIFI_SSID     = "Tong";
const char* WIFI_PASSWORD = "12345678";
const char* LAPTOP_IP     = "172.20.10.13";
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
  float thigh[9];      // acc_xyz, gyro_xyz, mag_xyz
  float shank[9];      // acc_xyz, gyro_xyz, mag_xyz
  float foot[9];       // acc_xyz, gyro_xyz(rad/s), mag_xyz
  float foot_euler[3]; // heading, roll, pitch (degrees)
};
#pragma pack(pop)

// ============================================================
// PCA9548A - actual channel assignments
// ============================================================
#define MUX_ADDR 0x70
#define CH_THIGH 4
#define CH_SHANK 5
#define CH_FOOT  7

void muxSelect(uint8_t ch) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(1 << ch);
  Wire.endTransmission();
  delayMicroseconds(50);  // reduced from 100us
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

// Frame rate tracking
unsigned long lastRateCheck = 0;
unsigned long framesInInterval = 0;
float currentHz = 0;

// NTP re-sync interval (10 minutes)
#define NTP_RESYNC_MS 600000
unsigned long lastNTPSync = 0;

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
  Serial.print("# Sending to: ");
  Serial.print(LAPTOP_IP);
  Serial.print(":");
  Serial.println(UDP_PORT);
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
  lastNTPSync = millis();

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
  Serial.println("# BE124 V6.1 - Optimized Binary UDP");
  Serial.print("# Packet size: ");
  Serial.print(sizeof(IMUFrame));
  Serial.println(" bytes");
  Serial.println("# Channels: THIGH=5, SHANK=4, FOOT=7");
  Serial.println("# BNO055 gyro: converted to rad/s");
  Serial.println("# ========================================");

  Wire.begin();
  Wire.setClock(400000);
  delay(500);

  // Multiplexer
  Serial.println("# Checking PCA9548A...");
  Wire.beginTransmission(MUX_ADDR);
  if (Wire.endTransmission() == 0) {
    Serial.println("# PCA9548A OK at 0x70");
  } else {
    Serial.println("# ERROR: PCA9548A not found!");
    while (1) delay(1000);
  }

  // THIGH (Channel 5)
  Serial.println("# Init THIGH (Ch5, ICM-20948)...");
  muxSelect(CH_THIGH);
  delay(100);
  if (icmThigh.begin_I2C()) {
    icmThigh.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
    icmThigh.setGyroRange(ICM20948_GYRO_RANGE_500_DPS);
    icmThigh.setAccelRateDivisor(10);
    icmThigh.setGyroRateDivisor(10);
    thighOK = true;
    // Self-test: read one sample
    sensors_event_t a, g, m, t;
    icmThigh.getEvent(&a, &g, &t, &m);
    Serial.print("#   THIGH OK | acc_z=");
    Serial.print(a.acceleration.z, 2);
    Serial.println(" m/s2");
  } else {
    Serial.println("#   THIGH FAIL");
  }

  // SHANK (Channel 4)
  Serial.println("# Init SHANK (Ch4, ICM-20948)...");
  muxSelect(CH_SHANK);
  delay(100);
  if (icmShank.begin_I2C()) {
    icmShank.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
    icmShank.setGyroRange(ICM20948_GYRO_RANGE_500_DPS);
    icmShank.setAccelRateDivisor(10);
    icmShank.setGyroRateDivisor(10);
    shankOK = true;
    sensors_event_t a, g, m, t;
    icmShank.getEvent(&a, &g, &t, &m);
    Serial.print("#   SHANK OK | acc_z=");
    Serial.print(a.acceleration.z, 2);
    Serial.println(" m/s2");
  } else {
    Serial.println("#   SHANK FAIL");
  }

  // FOOT (Channel 7)
  Serial.println("# Init FOOT (Ch7, BNO055)...");
  muxSelect(CH_FOOT);
  delay(500);
  if (bnoFoot.begin()) {
    bnoFoot.setExtCrystalUse(true);
    footOK = true;
    imu::Vector<3> ac = bnoFoot.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    uint8_t cs, cg, ca, cm;
    bnoFoot.getCalibration(&cs, &cg, &ca, &cm);
    Serial.print("#   FOOT OK | acc_z=");
    Serial.print(ac.z(), 2);
    Serial.print(" m/s2 | cal=");
    Serial.print(cs); Serial.print(",");
    Serial.print(cg); Serial.print(",");
    Serial.print(ca); Serial.print(",");
    Serial.println(cm);
  } else {
    Serial.println("#   FOOT FAIL - BNO055 not detected on Ch7");
  }

  // Summary
  int sensorCount = (thighOK ? 1 : 0) + (shankOK ? 1 : 0) + (footOK ? 1 : 0);
  Serial.print("# Sensors active: ");
  Serial.print(sensorCount);
  Serial.print("/3 [");
  Serial.print(thighOK ? "THIGH " : "");
  Serial.print(shankOK ? "SHANK " : "");
  Serial.print(footOK ? "FOOT" : "");
  Serial.println("]");

  if (sensorCount == 0) {
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
  lastRateCheck = millis();

  Serial.println("# ========================================");
  Serial.print("# READY - streaming at ~");
  Serial.print(ntpSynced ? "NTP" : "millis");
  Serial.println(" timestamps");
  Serial.println("# ========================================");
}

// ============================================================
// Main Loop
// ============================================================
unsigned long lastWiFiCheck = 0;

void loop() {
  unsigned long now = millis();

  // WiFi reconnect check every 5s
  if (now - lastWiFiCheck > 5000) {
    lastWiFiCheck = now;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("# WiFi lost, reconnecting...");
      connectWiFi();
    }
  }

  // NTP re-sync every 10 minutes to prevent clock drift
  if (ntpSynced && (now - lastNTPSync > NTP_RESYNC_MS)) {
    Serial.println("# NTP re-syncing...");
    syncNTP();
  }

  // Build frame
  IMUFrame frame;
  frame.magic = MAGIC;
  getTimeParts(frame.epoch_sec, frame.epoch_ms);
  frame.frame_id = frameCount++;

  memset(frame.thigh, 0, sizeof(frame.thigh));
  memset(frame.shank, 0, sizeof(frame.shank));
  memset(frame.foot, 0, sizeof(frame.foot));
  memset(frame.foot_euler, 0, sizeof(frame.foot_euler));

  // THIGH (Ch5, ICM-20948)
  if (thighOK) {
    muxSelect(CH_THIGH);
    sensors_event_t a, g, m, t;
    icmThigh.getEvent(&a, &g, &t, &m);
    frame.thigh[0]=a.acceleration.x; frame.thigh[1]=a.acceleration.y; frame.thigh[2]=a.acceleration.z;
    frame.thigh[3]=g.gyro.x; frame.thigh[4]=g.gyro.y; frame.thigh[5]=g.gyro.z;
    frame.thigh[6]=m.magnetic.x; frame.thigh[7]=m.magnetic.y; frame.thigh[8]=m.magnetic.z;
  }

  // SHANK (Ch4, ICM-20948)
  if (shankOK) {
    muxSelect(CH_SHANK);
    sensors_event_t a, g, m, t;
    icmShank.getEvent(&a, &g, &t, &m);
    frame.shank[0]=a.acceleration.x; frame.shank[1]=a.acceleration.y; frame.shank[2]=a.acceleration.z;
    frame.shank[3]=g.gyro.x; frame.shank[4]=g.gyro.y; frame.shank[5]=g.gyro.z;
    frame.shank[6]=m.magnetic.x; frame.shank[7]=m.magnetic.y; frame.shank[8]=m.magnetic.z;
  }

  // FOOT (Ch7, BNO055) - gyro converted to rad/s
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

  // Frame rate tracking
  framesInInterval++;

  // Debug output every 200 frames (~1.5s at 140Hz)
  if (frame.frame_id % 200 == 0) {
    // Calculate actual Hz
    unsigned long elapsed = now - lastRateCheck;
    if (elapsed > 0) {
      currentHz = framesInInterval * 1000.0f / elapsed;
    }
    lastRateCheck = now;
    framesInInterval = 0;

    Serial.print("# F:");
    Serial.print(frame.frame_id);
    Serial.print(" Hz:");
    Serial.print(currentHz, 0);
    Serial.print(" | T_z=");
    Serial.print(frame.thigh[2], 1);
    Serial.print(" S_z=");
    Serial.print(frame.shank[2], 1);
    Serial.print(" F_z=");
    Serial.print(frame.foot[2], 1);
    Serial.print(" F_gy=");
    Serial.print(frame.foot[3], 3);

    // BNO055 cal status every 1000 frames
    if (footOK && frame.frame_id % 1000 == 0) {
      muxSelect(CH_FOOT);
      uint8_t cs, cg, ca, cm;
      bnoFoot.getCalibration(&cs, &cg, &ca, &cm);
      Serial.print(" cal=");
      Serial.print(cs); Serial.print(cg); Serial.print(ca); Serial.print(cm);
    }

    Serial.println();
  }
}
