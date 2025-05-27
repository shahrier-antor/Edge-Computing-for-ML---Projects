#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model and input image headers
#include "model.h"
#include "test_image_live.h"  // contains image_data[]

/* -------------------------------------------------------
   Global Variables for TensorFlow Lite Micro
--------------------------------------------------------*/

// Pull in all TFLM ops — for minimal build, include only required ops
tflite::AllOpsResolver opResolver;

// Model and Interpreter
const tflite::Model* modelPointer = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

// Allocate static memory for tensor arena
constexpr int kTensorArenaSize = 140 * 1024;
byte tensorArena[kTensorArenaSize] __attribute__((aligned(16)));

// Labels for output classes
const char* kGestureLabels[] = {
  "rock",
  "paper",
  "scissors"
};
#define kNumGestures (sizeof(kGestureLabels) / sizeof(kGestureLabels[0]))

/* -------------------------------------------------------
   Setup Function: Initialize model, interpreter, tensors
--------------------------------------------------------*/
void setup() {
  Serial.begin(9600);
  while (!Serial);  // Wait for serial monitor

  // Load the TFLite model
  modelPointer = tflite::GetModel(model);
  if (modelPointer->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  interpreter = new tflite::MicroInterpreter(modelPointer, opResolver, tensorArena, kTensorArenaSize);

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Tensor allocation failed.");
    while (1);
  }

  // Get pointers to input/output tensors
  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);
}

/* -------------------------------------------------------
   Loop Function: Perform inference on test image
--------------------------------------------------------*/
void loop() {
  // Load flattened grayscale input (32×32)
  for (int i = 0; i < (32 * 32); i++) {
    inputTensor->data.uint8[i] = image_data[i];
  }

  // Run inference
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("❌ Inference failed!");
    while (1);
    return;
  }

  // Print output probabilities
  Serial.println("🔍 Inference Result:");
  for (int i = 0; i < kNumGestures; i++) {
    Serial.print(kGestureLabels[i]);
    Serial.print(": ");
    Serial.println(outputTensor->data.uint8[i]);
  }

  Serial.println();
  delay(20000);  // Wait 20 seconds before next inference
}
