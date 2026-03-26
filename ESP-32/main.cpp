#include <Arduino.h>
#include "FastLED.h"
#include "stdio.h"
#include "vector"
#include "types.h"

#define LED_PIN 25
#define COLOR_ORDER GRB
#define BRIGHTNESS 128
#define NUM_LEDS 150
#define UPDATE_TIME 2000 // in milliseconds
#define STEPS 70
#define FADE_TYPE FADE_TYPES::STEPPING
#define CLR_SELECTION CLR_SELECTION_METHOD::WEIGHTED

CRGB leds[NUM_LEDS];
CRGB currentColor = CRGB::Black;

#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled!
#endif

BluetoothSerial SerialBT;


void setup() {
  delay(3000); // Startup safety delay
  FastLED.addLeds<WS2812B, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);
  Serial.begin(115200);

  SerialBT.begin("ESP-32_Mood"); //Bluetooth device name
  Serial.println("Device active. Ready to pair!");
}

void blendSwitch(CRGB switchColor, CRGB newcolor)
{

  int TIME_STEP = (int)(UPDATE_TIME/STEPS);
  for (int i = 0; i < STEPS; i++)
  {
    CRGB intermediateColor = blend(switchColor, newcolor, (i+1)*255/STEPS);
    fill_solid(leds, NUM_LEDS, intermediateColor);
    FastLED.show();
    delay(TIME_STEP-1);
  }
  currentColor = newcolor;
}

void stepSwitch(CRGB switchColor, CRGB newcolor)
{
int TIME_PER_LED = UPDATE_TIME / NUM_LEDS; 

  for (int i = 0; i < NUM_LEDS; i++)
  {
    leds[i] = newcolor;
    FastLED.show();
    delay(TIME_PER_LED);
  }

  currentColor = newcolor; 
}

void switchLED(CRGB switchColor, CRGB newcolor)
{
  switch (FADE_TYPE)
  {
  case FADE_TYPES::BLEND:
    blendSwitch(switchColor, newcolor);
    break;
  case FADE_TYPES::STEPPING:
    stepSwitch(switchColor, newcolor);
    break;
  default:
    break;
  }
}

int myFunction(int, int);

CRGB getColorCentroid(CRGB vec1, CRGB vec2, CRGB vec3)
{
  uint8_t r = (vec1.r + vec2.r + vec3.r) /3;
  uint8_t g = (vec1.g + vec2.g + vec3.g) /3;
  uint8_t b = (vec1.b + vec2.b + vec3.b) /3;
  return {r, g, b};
}

CRGB getColorEncoding(EMOTIONS colorIdx)
{
  switch (colorIdx)
  {
  case EMOTIONS::ANG: return CRGB{255, 0, 0};
  case EMOTIONS::DIS: return CRGB{255, 78, 3};
  case EMOTIONS::FEA: return CRGB{128, 0, 190};
  case EMOTIONS::HAP: return CRGB{242, 187, 5};
  case EMOTIONS::NEU: return CRGB{245, 245, 245};
  case EMOTIONS::SAD: return CRGB{0, 0, 196};
  case EMOTIONS::CAL: return CRGB{162, 162, 252};
  case EMOTIONS::SUR: return CRGB{50, 255, 140};
  default: return CRGB{0, 0, 0}; 
  }
}

void swap(int* a, int* b){
  int t = *a;
  *a = *b;
  *b = t;
}

void getEmtn(float* probabilityArray, int* topThreeEmtns)
{
  for (int i = 0; i <= 7; i++)
  {
    for (int j = 0; j <=2; j++)
    {
      if (topThreeEmtns[j] < 0) {topThreeEmtns[j] = i; break;}
      if (probabilityArray[i] > probabilityArray[topThreeEmtns[j]])
      {
        for (int k=2; k>j; k--)
        {
          topThreeEmtns[k] = topThreeEmtns[k-1];
        }
        topThreeEmtns[j] = i;
        break;
      }
    }
  }
} 

void BlinkLED(){
  static bool lightOn = false;
  static uint8_t hue = 0;
  lightOn = !lightOn;

  if(lightOn){
    switchLED(currentColor, CRGB(hue, 255 - hue, 255));
    hue +=32;
  } else {
    //fill_solid(leds, NUM_LEDS, CRGB::Black);
  }
  FastLED.show();
  delay(lightOn ? 2000 : 900);
}


CRGB getWeightedColor(CRGB vec1, CRGB vec2, CRGB vec3, float p1, float p2, float p3)
{
  // Square the probabilities to amplify the dominant emotion
  float p1_sq = p1 * p1;
  float p2_sq = p2 * p2;
  float p3_sq = p3 * p3;

  float sum = p1_sq + p2_sq + p3_sq;
  
  // Avoids division by zero
  if (sum <= 0.001f) return CRGB::Black; 

  float w1 = p1_sq / sum;
  float w2 = p2_sq / sum;
  float w3 = p3_sq / sum;

  uint8_t r = (uint8_t)(vec1.r * w1 + vec2.r * w2 + vec3.r * w3);
  uint8_t g = (uint8_t)(vec1.g * w1 + vec2.g * w2 + vec3.g * w3);
  uint8_t b = (uint8_t)(vec1.b * w1 + vec2.b * w2 + vec3.b * w3);

  return {r, g, b};
}
void loop() {
  String payload = "";
  payload.reserve(100); // A bit more than the expected payload size to avoid fragmentation
  if (SerialBT.available())
  {
    payload = SerialBT.readStringUntil('X');
  }
  if (payload.length() > 0)
  {
    Serial.println("Received payload: " + payload);
    float probabilityArray[8];
    int topThreeEmtns[3] = {-1, -1, -1};
    sscanf(payload.c_str(), "%f,%f,%f,%f,%f,%f,%f,%f", 
      &probabilityArray[0], &probabilityArray[1], &probabilityArray[2], &probabilityArray[3], 
      &probabilityArray[4], &probabilityArray[5], &probabilityArray[6], &probabilityArray[7]);
           
    getEmtn(probabilityArray, topThreeEmtns);    
    CRGB color1 = getColorEncoding(static_cast<EMOTIONS>(topThreeEmtns[0]));
    CRGB color2 = getColorEncoding(static_cast<EMOTIONS>(topThreeEmtns[1]));
    CRGB color3 = getColorEncoding(static_cast<EMOTIONS>(topThreeEmtns[2]));

    CRGB newColor = CRGB::Black;
    switch (CLR_SELECTION)
    {
    case CLR_SELECTION_METHOD::CENTROID:
      newColor = getColorCentroid(color1, color2, color3);
      break;
    case CLR_SELECTION_METHOD::WEIGHTED:
      newColor = getWeightedColor(color1, color2, color3, 
        probabilityArray[topThreeEmtns[0]], probabilityArray[topThreeEmtns[1]], probabilityArray[topThreeEmtns[2]]);
      break;
    case CLR_SELECTION_METHOD::ONE_COLOR:
      newColor = color1;
      break;
    default:
      break;
    }  
    switchLED(currentColor, newColor);
  }
  delay(100);
}