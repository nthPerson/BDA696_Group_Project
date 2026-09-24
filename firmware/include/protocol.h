// FormCoach BLE / serial protocol — single source of truth on the firmware side.
// Mirrored by src/formcoach/io/protocol.py; tests/test_protocol.py checks the FC_* constants,
// UUIDs and command codes below against the Python copy. Change both together.
#pragma once
#include <stdint.h>

// ---- Identity ------------------------------------------------------------------------------
#define FC_DEVICE_NAME_PREFIX "FormCoach-"
#define FC_SERVICE_UUID "7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001"
#define FC_IMU_CHAR_UUID "7a0c0002-4e1e-4b9a-9a1c-0f0c0c0c0001"      // notify: sample batches
#define FC_CONTROL_CHAR_UUID "7a0c0003-4e1e-4b9a-9a1c-0f0c0c0c0001"  // write: 1-byte commands
#define FC_STATUS_CHAR_UUID "7a0c0004-4e1e-4b9a-9a1c-0f0c0c0c0001"   // read/notify: status
#define FC_REQUESTED_MTU 185

// ---- Sampling and gate constants (keep identical to protocol.py / model/preprocess.json) ----
#define FC_SAMPLE_RATE_HZ 50
#define FC_WINDOW_SAMPLES 100         // 2 s
#define FC_WINDOW_STRIDE_SAMPLES 25   // gate inference every 0.5 s
#define FC_BATCH_SAMPLES 5            // samples per notification (100 ms)
#define FC_GATE_ON_WINDOWS 2          // consecutive active windows to open (1.0 s)
#define FC_GATE_OFF_WINDOWS 12        // consecutive idle windows to close (6 s)

// ---- Commands (laptop -> wearable, control characteristic) ---------------------------------
enum FcCommand : uint8_t {
  FC_CMD_LED_GOOD_REP = 0x01,
  FC_CMD_LED_FAULT = 0x02,
  FC_CMD_START_SESSION = 0x03,
  FC_CMD_STOP_SESSION = 0x04,
  FC_CMD_CALIBRATE = 0x05,
  FC_CMD_PING = 0x10,
};

// ---- Sample flags ---------------------------------------------------------------------------
enum FcFlag : uint8_t {
  FC_FLAG_GATE_STATE = 1 << 0,
  FC_FLAG_SESSION_ACTIVE = 1 << 1,
  FC_FLAG_BUTTON_PRESSED = 1 << 2,
  FC_FLAG_CALIBRATED = 1 << 3,
};

// ---- Wire layout (little-endian, packed) ---------------------------------------------------
#pragma pack(push, 1)
struct FcSample {
  uint32_t t_ms;  // device millis at sample
  int16_t ax, ay, az;  // raw accel LSB, +-8 g  (4096 LSB/g)
  int16_t gx, gy, gz;  // raw gyro LSB,  +-1000 dps (32.8 LSB/dps)
  uint8_t flags;  // FcFlag bits
  uint16_t seq;   // rolling sample counter for drop detection
};

struct FcBatchHeader {
  uint16_t session_id;
  uint8_t n;  // samples that follow (<= FC_BATCH_SAMPLES)
  uint8_t reserved;
};

struct FcBatch {
  FcBatchHeader header;
  FcSample samples[FC_BATCH_SAMPLES];
};
#pragma pack(pop)

#define FC_SAMPLE_SIZE 19
#define FC_BATCH_HEADER_SIZE 4
#define FC_MAX_BATCH_SIZE (FC_BATCH_HEADER_SIZE + FC_BATCH_SAMPLES * FC_SAMPLE_SIZE)  // 99

static_assert(sizeof(FcSample) == FC_SAMPLE_SIZE, "FcSample must be 19 packed bytes");
static_assert(sizeof(FcBatchHeader) == FC_BATCH_HEADER_SIZE, "FcBatchHeader must be 4 bytes");
static_assert(sizeof(FcBatch) == FC_MAX_BATCH_SIZE, "FcBatch must be 99 bytes");
static_assert(FC_MAX_BATCH_SIZE <= FC_REQUESTED_MTU - 3, "batch must fit one notification");

// ---- Status characteristic payload (read/notify) -------------------------------------------
#pragma pack(push, 1)
struct FcStatus {
  uint8_t fw_major, fw_minor, fw_patch;
  uint16_t session_id;
  uint8_t gate_state;   // 0 idle, 1 active
  uint32_t uptime_s;
  uint8_t calibrated;   // 1 when a gravity/gyro-bias calibration is stored in NVS
  uint8_t session_active;
};
#pragma pack(pop)
#define FC_STATUS_SIZE 12
static_assert(sizeof(FcStatus) == FC_STATUS_SIZE, "FcStatus must be 12 packed bytes");

// ---- Energy gate (laptop `--gate energy` and firmware v1 use the same threshold) ------------
// var(|a|) over the 2 s window, in (m/s^2)^2; balanced-accuracy fit on RecoFit (ADR-0019).
#define FC_GATE_ENERGY_THRESHOLD_MS2SQ 3.0f

// Dev-mode serial CSV (one line per sample, 115200 baud):
#define FC_SERIAL_CSV_HEADER "t_ms,ax,ay,az,gx,gy,gz,flags,seq"
// Serial control: one ASCII character per line mirrors the BLE command codes (protocol.py).
#define FC_SERIAL_CMD_LED_GOOD_REP 'G'   // FC_CMD_LED_GOOD_REP
#define FC_SERIAL_CMD_LED_FAULT 'F'      // FC_CMD_LED_FAULT
#define FC_SERIAL_CMD_START_SESSION 'S'  // FC_CMD_START_SESSION
#define FC_SERIAL_CMD_STOP_SESSION 'X'   // FC_CMD_STOP_SESSION
#define FC_SERIAL_CMD_CALIBRATE 'C'      // FC_CMD_CALIBRATE
#define FC_SERIAL_CMD_PING 'P'           // FC_CMD_PING
