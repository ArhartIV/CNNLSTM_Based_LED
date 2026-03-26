## CNN/LSTM powered Emotion-based LED (Work in Progress)
### Description
An end-to-end, real-time speech emotion recognition system that translates human emotions into ambient LED lighting. I initially started the project as a away to understand the inner workings of Neural Networks and get experience with their internals. This model was built completely from scratch using only NumPy. I manually derived and implemented the backpropagation calculus, custom layers (CNN, LSTM, Attention), and optimizers (Adam) without relying on frameworks like TensorFlow or PyTorch. I later learned how to integrate it dynamically with the microcontroller to adjust the LED lighting.


### Structure 
The system operates in a continuous real-time loop:

1. **Audio Capture:** Records ~4-second audio snippets in real-time.
2. **Feature Extraction:** Processes raw audio into Mel Spectrograms (with deltas and delta-deltas) using custom-built FFT, Mel filters, and Hamming windows.
3. **AI Inference:** Propagates the mel spectrograms through the custom CNN/LSTM model to generate probability distributions across 8 distinct emotions.
4. **Hardware Execution:** Transmits the emotion probabilities via Bluetooth to an ESP32 microcontroller, which calculates the mathematical centroid of the top 3 dominant emotions and smoothly crossfades a WS2812B LED strip to the resulting color.

### Technical data:
* **Machine Learning:** Python, NumPy, Custom Math/Backprop
* **Audio Processing:** PyAudio, Custom FFT/Mel-filter algorithms
* **Hardware:** C++, ESP32, WS2812B LEDs, FastLED Library, Classic Bluetooth (SPP)

### Model
* **Feature Extractor (CNN):** A series of 2D Convolutions, Batch Normalization, and GELU activations to extract baseline acoustic patterns. It includes a custom Spatial Attribution layer to weigh important frequency bands.
* **Temporal Sequence (LSTM):** A bidirectional LSTM module processes the CNN outputs to understand the time-relation and sequential context of the previously extracted audio features.
* **Classifier (Dense):** A fully connected Dense decoder that maps the resulting data from the LSTM into an 8-class softmax output (Anger, Disgust, Fear, Happiness, Neutral, Sadness, Calm, Surprise).

I first uses a series of convolution layers with the increasing kernel count to get general features and patterns from mel spectrograms. Then, the result is passed through LSTM nodule that is meant to extract time-relation and importance of previously found features. The Decoder is used in the end to map the result to 8 emotions.<br>
This model achieved 77.91% test accuracy on the used dataset. <br>
![Test Set Confusion Matrix](/imgs/Test_Confusion_Matrix_HighRes.png "Test Set Confusion Matrix")
![Validation Set Confusion Matrix](/imgs/Validation_Confusion_Matrix_HighRes.png "Validation Set Confusion Matrix")

The Model was later fine tuned to my voice using the recordings


### Sources
Mel extraction algorithm baseline inspired by Jonathan Hui's Audio Processing Guide.
Base training dataset provided by Hugging Face (stapesai/ssi-speech-emotion-recognition).

### Further Development
I will later add videos/GIFs showcasing different modes for the ESP32 and more statistical visualizations of the model. After finishing the Fine-tuning for INMP441, i also plan to develop it into stand-alone PCB board. 