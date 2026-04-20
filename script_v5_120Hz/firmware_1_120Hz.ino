/*
 * BE124 Ankle Exoskeleton V5 - Complete Frame UDP
 * =================================================
 * Key change from V4: reads all 3 IMUs first, then sends
 * ONE complete UDP packet with all data. Zero empty fields.
 *
 * Board: Adafruit ESP32-S3 Feather
 * Multiplexer: PCA9548A (0x70)
 * Sensors:
 *   Ch0: ICM-20948 (THIGH)
 *   Ch1: ICM-20948 (SHANK)
 *   Ch2: BNO055    (FOOT)
 *
 * Output format (one line per frame):
 *   timestamp,thigh_acc_x,...,thigh_mag_z,shank_acc_x,...,shank_mag_z,foot_acc_x,...,foot_euler_z
 *
 * Wiring:
 *   ESP32 SDA (GPIO 3) -> PCA9548A SDA
 *   ESP32 SCL (GPIO 4) -> PCA9548A SCL
 *   ESP32 3V           -> PCA9548A VIN + all IMU VINs
 *   ESP32 GND          -> PCA9548A GND + all IMU GNDs
 *   PCA9548A Ch0       -> ICM-20948 #1 (THIGH)
 *   PCA9548A Ch1       -> ICM-20948 #2 (SHANK)
 *   PCA9548A Ch2       -> BNO055       (FOOT)
 *
 * Libraries: Adafruit ICM20X, Adafruit BNO055,
 *            Adafruit Unified Sensor, Adafruit BusIO
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
const char* WIFI_SSID     = "Suckest-Wifi-in-Boston";
const char* WIFI_PASSWORD = "z3234220148";
const char* LAPTOP_IP     = "192.168.1.164";
const int   UDP_PORT      = 12345;

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

String getTimestamp() {
  if (!ntpSynced) return String(millis());
  unsigned long elapsed = millis() - ntpSyncMillis;
  unsigned long totalUs = ntpSyncMicros + (elapsed % 1000) * 1000;
  time_t epoch = ntpSyncEpoch + (elapsed / 1000) + (totalUs / 1000000);
  unsigned long ms = (totalUs % 1000000) / 1000;
  char buf[20];
  snprintf(buf, sizeof(buf), "%lu.%03lu", (unsigned long)epoch, ms);
  return String(buf);
}

// ============================================================
// Setup
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n# ========================================");
  Serial.println("# BE124 V5 - Complete Frame UDP");
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

  // WiFi
  if (!connectWiFi()) {
    Serial.println("# WARNING: No WiFi. Data only on Serial.");
  }

  // NTP
  if (WiFi.status() == WL_CONNECTED) {
    ntpSynced = syncNTP();
    if (!ntpSynced) Serial.println("# WARNING: NTP failed, using millis()");
  }

  // UDP
  udp.begin(UDP_PORT);

  // Print CSV header
  Serial.println("# ========================================");
  Serial.println("# CSV Header:");
  Serial.println("# timestamp,thigh_acc_x,thigh_acc_y,thigh_acc_z,thigh_gyro_x,thigh_gyro_y,thigh_gyro_z,thigh_mag_x,thigh_mag_y,thigh_mag_z,shank_acc_x,shank_acc_y,shank_acc_z,shank_gyro_x,shank_gyro_y,shank_gyro_z,shank_mag_x,shank_mag_y,shank_mag_z,foot_acc_x,foot_acc_y,foot_acc_z,foot_gyro_x,foot_gyro_y,foot_gyro_z,foot_mag_x,foot_mag_y,foot_mag_z,foot_euler_x,foot_euler_y,foot_euler_z");
  Serial.println("# ========================================");
  Serial.println("# READY - streaming complete frames");
}

// ============================================================
// Main Loop
// ============================================================
unsigned long lastSample = 0;
unsigned long lastWiFiCheck = 0;
// Use a char buffer for building the frame string efficiently
char frameBuf[512];

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

  // No delay - run as fast as possible
  // The bottleneck is I2C reads (~3-5ms for all 3 sensors at 400kHz)

  // ---- Read ALL sensors first, THEN send ----

  // Timestamp for this frame (taken before reads for consistency)
  String ts = getTimestamp();

  // THIGH
  float ta_x=0,ta_y=0,ta_z=0, tg_x=0,tg_y=0,tg_z=0, tm_x=0,tm_y=0,tm_z=0;
  if (thighOK) {
    muxSelect(CH_THIGH);
    sensors_event_t a, g, m, t;
    icmThigh.getEvent(&a, &g, &t, &m);
    ta_x=a.acceleration.x; ta_y=a.acceleration.y; ta_z=a.acceleration.z;
    tg_x=g.gyro.x; tg_y=g.gyro.y; tg_z=g.gyro.z;
    tm_x=m.magnetic.x; tm_y=m.magnetic.y; tm_z=m.magnetic.z;
  }

  // SHANK
  float sa_x=0,sa_y=0,sa_z=0, sg_x=0,sg_y=0,sg_z=0, sm_x=0,sm_y=0,sm_z=0;
  if (shankOK) {
    muxSelect(CH_SHANK);
    sensors_event_t a, g, m, t;
    icmShank.getEvent(&a, &g, &t, &m);
    sa_x=a.acceleration.x; sa_y=a.acceleration.y; sa_z=a.acceleration.z;
    sg_x=g.gyro.x; sg_y=g.gyro.y; sg_z=g.gyro.z;
    sm_x=m.magnetic.x; sm_y=m.magnetic.y; sm_z=m.magnetic.z;
  }

  // FOOT
  float fa_x=0,fa_y=0,fa_z=0, fg_x=0,fg_y=0,fg_z=0, fm_x=0,fm_y=0,fm_z=0;
  float fe_x=0, fe_y=0, fe_z=0;
  if (footOK) {
    muxSelect(CH_FOOT);
    imu::Vector<3> ac = bnoFoot.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
    imu::Vector<3> gy = bnoFoot.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    imu::Vector<3> mg = bnoFoot.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
    imu::Vector<3> eu = bnoFoot.getVector(Adafruit_BNO055::VECTOR_EULER);
    fa_x=ac.x(); fa_y=ac.y(); fa_z=ac.z();
    fg_x=gy.x()/57.2958; fg_y=gy.y()/57.2958; fg_z=gy.z()/57.2958;
    fm_x=mg.x(); fm_y=mg.y(); fm_z=mg.z();
    fe_x=eu.x(); fe_y=eu.y(); fe_z=eu.z();
  }

  // ---- Build one complete frame string ----
  int len = snprintf(frameBuf, sizeof(frameBuf),
    "%s,"
    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
    "%.2f,%.2f,%.2f",
    ts.c_str(),
    ta_x,ta_y,ta_z, tg_x,tg_y,tg_z, tm_x,tm_y,tm_z,
    sa_x,sa_y,sa_z, sg_x,sg_y,sg_z, sm_x,sm_y,sm_z,
    fa_x,fa_y,fa_z, fg_x,fg_y,fg_z, fm_x,fm_y,fm_z,
    fe_x,fe_y,fe_z
  );

  // ---- Send ONE packet with all data ----
  if (WiFi.status() == WL_CONNECTED) {
    udp.beginPacket(LAPTOP_IP, UDP_PORT);
    udp.write((const uint8_t*)frameBuf, len);
    udp.endPacket();
  }

  // Also print to Serial
  Serial.println(frameBuf);
}
