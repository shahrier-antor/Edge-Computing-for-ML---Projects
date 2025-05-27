#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino_OV767X.h>

#include "model.h"  // Your quantized TFLite model as a byte array

#define IMG_W 32
#define IMG_H 32
#define IMG_PIXELS (IMG_W * IMG_H)

const char* CLASS_LABELS[] = {"rock", "paper", "scissors"};
#define NUM_CLASSES (sizeof(CLASS_LABELS) / sizeof(CLASS_LABELS[0]))

// Grayscale camera frame buffer
unsigned short camera_frame[160 * 120];

// ===== Memory Check Function =====
#ifdef __arm__
extern "C" char* sbrk(int incr);
int getFreeMemory() {
  char top;
  return &top - reinterpret_cast<char*>(sbrk(0));
}
#else
extern "C" char __heap_start, *__brkval;
int getFreeMemory() {
  if (__brkval == 0) return ((int)&getFreeMemory) - ((int)&__heap_start);
  return ((int)&getFreeMemory) - ((int)__brkval);
}
#endif

void initializeCamera() {
  Serial.println("[INFO] Setting up camera...");

  if (!Camera.begin(QQVGA, GRAYSCALE, 1)) {
    Serial.println("[ERROR] Camera init failed.");
    while (true);
  }

  Serial.println("[OK] Camera is ready.");
  Serial.print("RAM remaining: ");
  Serial.println(getFreeMemory());
}

void performInference(uint8_t* input_image) {
  Serial.println("[RUN] Beginning inference process...");

  const tflite::Model* model_ref = tflite::GetModel(model);
  if (model_ref->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model schema mismatch.");
    return;
  }

  tflite::AllOpsResolver resolver;

  constexpr int kArenaSize = 60 * 1024;
  static uint8_t tensor_arena[kArenaSize] __attribute__((aligned(16)));

  tflite::MicroInterpreter interpreter(model_ref, resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("[FAIL] Could not allocate tensors.");
    return;
  }

  TfLiteTensor* input_tensor = interpreter.input(0);
  for (int i = 0; i < IMG_PIXELS; i++) {
    input_tensor->data.uint8[i] = input_image[i];
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    Serial.println("[FAIL] Interpreter invoke failed.");
    return;
  }

  TfLiteTensor* output_tensor = interpreter.output(0);
  Serial.println("[RESULT] Prediction probabilities:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(" → ");
    Serial.print(CLASS_LABELS[i]);
    Serial.print(": ");
    Serial.println(output_tensor->data.uint8[i]);
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("=== Rock-Paper-Scissors Classifier ===");
  initializeCamera();
}

void loop() {
  if (Serial.read() == 'c') {
    Serial.println("[CAPTURE] Taking photo...");

    Camera.readFrame(camera_frame);
    Serial.println("[OK] Frame captured.");

    uint8_t downsampled_img[IMG_PIXELS];
    const int x_step = 160 / IMG_W;
    const int y_step = 120 / IMG_H;

    for (int y = 0; y < IMG_H; y++) {
      for (int x = 0; x < IMG_W; x++) {
        int src_index = (y * y_step) * 160 + (x * x_step);
        downsampled_img[y * IMG_W + x] = camera_frame[src_index] >> 8;
      }
    }

    Serial.println("[INFO] Image resized. Running model...");
    performInference(downsampled_img);
    Serial.println("[WAIT] Press 'c' to classify again.");
  }
}
