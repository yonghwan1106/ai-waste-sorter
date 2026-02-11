/*
 * AI 스마트 분리수거 로봇 - Arduino 서보 게이트 제어
 *
 * 회로 연결:
 *   - 서보1 (방향): D9  (좌/중/우 방향 전환)
 *   - 서보2 (게이트): D10 (게이트 개폐)
 *   - 초음파 Trig: D7
 *   - 초음파 Echo: D8
 *   - LED 플라스틱: D2 (주황)
 *   - LED 캔:       D3 (흰색)
 *   - LED 종이:     D4 (갈색)
 *   - LED 유리:     D5 (초록)
 *   - LED 비닐/일반: D6 (노랑)
 *
 * 시리얼 프로토콜:
 *   수신: "0"~"5" (클래스 ID) 또는 "R" (리셋)
 *   송신: "OK:{class_id}" 또는 "RESET"
 */

#include <Servo.h>

// === 핀 설정 ===
#define SERVO1_PIN 9    // 방향 서보
#define SERVO2_PIN 10   // 게이트 서보
#define TRIG_PIN   7    // 초음파 Trig
#define ECHO_PIN   8    // 초음파 Echo
#define LED_PINS   {2, 3, 4, 5, 6}  // 5개 LED

// === 서보 각도 설정 ===
// 서보1: 분류 방향 (0=좌, 90=중앙, 180=우)
// 서보2: 게이트 (0=닫힘, 90=열림)
const int SERVO1_ANGLES[] = {0, 0, 90, 90, 180, 180};  // 클래스별 방향
const int SERVO2_OPEN = 90;    // 게이트 열림
const int SERVO2_CLOSED = 0;   // 게이트 닫힘
const int SERVO1_CENTER = 90;  // 초기 위치

// === 타이밍 ===
const unsigned long GATE_OPEN_TIME = 2000;  // 게이트 열림 시간 (ms)
const unsigned long DETECT_INTERVAL = 200;  // 초음파 감지 간격 (ms)
const float DETECT_DISTANCE = 15.0;         // 물체 감지 거리 (cm)

// === 전역 변수 ===
Servo servo1, servo2;
int ledPins[] = LED_PINS;
bool isProcessing = false;
unsigned long lastDetectTime = 0;

void setup() {
  Serial.begin(9600);

  // 서보 초기화
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  resetGate();

  // 초음파 센서
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // LED 초기화
  for (int i = 0; i < 5; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }

  // 시작 표시 (LED 순차 점등)
  for (int i = 0; i < 5; i++) {
    digitalWrite(ledPins[i], HIGH);
    delay(100);
    digitalWrite(ledPins[i], LOW);
  }

  Serial.println("READY");
}

void loop() {
  // 시리얼 명령 수신
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "R") {
      // 리셋 명령
      resetGate();
      Serial.println("RESET");
    }
    else if (cmd.length() == 1 && cmd[0] >= '0' && cmd[0] <= '5') {
      // 분류 명령 (0~5)
      int classId = cmd.toInt();
      sortWaste(classId);
      Serial.print("OK:");
      Serial.println(classId);
    }
    else if (cmd == "D") {
      // 거리 측정 요청
      float dist = measureDistance();
      Serial.print("DIST:");
      Serial.println(dist, 1);
    }
  }

  // 초음파 센서로 물체 감지 (주기적)
  unsigned long now = millis();
  if (!isProcessing && (now - lastDetectTime > DETECT_INTERVAL)) {
    float distance = measureDistance();
    if (distance > 0 && distance < DETECT_DISTANCE) {
      // 물체 감지됨 -> Python에 알림
      Serial.print("DETECTED:");
      Serial.println(distance, 1);
    }
    lastDetectTime = now;
  }
}

/**
 * 쓰레기 분류 실행
 * 1. 해당 LED 점등
 * 2. 서보1을 분류 방향으로 회전
 * 3. 서보2로 게이트 열기
 * 4. 대기 후 게이트 닫기
 */
void sortWaste(int classId) {
  if (classId < 0 || classId > 5) return;
  isProcessing = true;

  // 모든 LED 끄기
  allLedOff();

  // 해당 분류 LED 점등
  // LED 매핑: 0,1->LED0 / 2->LED1 / 3->LED2 / 4->LED3 / 5->LED4
  int ledIndex;
  if (classId <= 1) ledIndex = 0;       // 플라스틱/캔
  else if (classId == 2) ledIndex = 1;  // 종이
  else if (classId == 3) ledIndex = 2;  // 유리
  else if (classId == 4) ledIndex = 3;  // 비닐
  else ledIndex = 4;                     // 일반쓰레기

  digitalWrite(ledPins[ledIndex], HIGH);

  // 서보1: 분류 방향으로 회전
  servo1.write(SERVO1_ANGLES[classId]);
  delay(500);  // 서보 이동 대기

  // 서보2: 게이트 열기
  servo2.write(SERVO2_OPEN);
  delay(GATE_OPEN_TIME);

  // 게이트 닫기 + LED 끄기
  servo2.write(SERVO2_CLOSED);
  delay(500);

  // 서보1 중앙으로 복귀
  servo1.write(SERVO1_CENTER);
  delay(300);

  // LED 끄기
  digitalWrite(ledPins[ledIndex], LOW);

  isProcessing = false;
}

/**
 * 게이트 초기 위치로 리셋
 */
void resetGate() {
  servo1.write(SERVO1_CENTER);
  servo2.write(SERVO2_CLOSED);
  allLedOff();
  isProcessing = false;
}

/**
 * 초음파 센서로 거리 측정 (cm)
 */
float measureDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);  // 30ms 타임아웃

  if (duration == 0) return -1.0;  // 타임아웃

  float distance = duration * 0.034 / 2.0;
  return distance;
}

/**
 * 모든 LED 끄기
 */
void allLedOff() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(ledPins[i], LOW);
  }
}
