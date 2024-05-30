# VuBot

## Description
VuBot is an application that combines speech and gesture recognition to interact with objects in a real-time video feed. Using a webcam, users can point at objects and issue voice commands to perform actions such as detecting individual objects, recognizing all objects in the scene, or querying the color of a specific object. VuBot leverages powerful libraries and models like MediaPipe for gesture detection, OpenCV for video processing, and OpenAI whisper for capturing and processing voice commands.

### Key Features
- Gesture Recognition: Detects gestures such as pointing, closed fist, and victory using MediaPipe.
- Speech Recognition: Processes voice commands to trigger actions like object detection and color recognition.
- Object Detection: Identifies objects in the video feed and draws bounding boxes around them.
- Color Recognition: Determines the color of objects by averaging the colors within the bounding boxes.
  
VuBot is designed to be intuitive and user-friendly, making it a versatile tool for various applications. 

## Models used
- **Gesture Recognition**: The application uses the [MediaPipe library](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer) to recognize gestures. 
- **Speech Recognition**: The application uses the [OpenAI whisper model](https://github.com/openai/whisper) to recognize speech.
- **Object Recognition**: The application uses the [Facebook DETR model](https://huggingface.co/facebook/detr-resnet-50) to recognize objects.

## Installation

**1** - First, clone the project from the repository and navigate to the project root:
```sh
git clone https://github.com/darmangerd/vubot.git

cd vubot
```
  
**2** - Next, install the project dependencies, preferably in a virtual environment. To do this, execute the following commands from the project root:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3** - Finally, run the project:
```sh
python app.py
```
- Be sure to have a functional microphone and webcam connected to your computer.

## Guide

| Gesture            | Trigger Word      | Output                         |
|--------------------|-------------------|--------------------------------|
| Pointing | **'object'**      | Return object's name           |
| Pointing | **'color'**       | Return object's color          |
| Closed Fist        | **'every item'**  | Highlight all detected objects | 

## Project Structure
- **app.py**: Main file to run the project.
- **evaluation_keys.py**: File containing the alternative keys method, used to evaluate the project (object and color names are manipulated).
- **evaluation_speech.py**: File containing the alternative speech method, used to evaluate the project (object and color names are manipulated).
- **requirements.txt**: File containing the project dependencies.
- **/images**: Folder containing the images saved for debugging purposes.
- **/utils**: Folder containing the utility functions used in the project.
  - **/utils/models**: Folder containing the gesture recognition model used in the project (mediapipe).
  - **/utils/main_evaluation.csv**: File containing the evaluation data for the project obtained during the evaluation phase.
- **/docs**: Folder containing the project documentation.


## Future Work
Future enhancements include developing a mobile version, improving audio speech handling, adding more interaction methods, integrating a large language model (LLM) for richer interactions, and implementing features to remember and locate specific objects.
