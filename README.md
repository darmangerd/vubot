# VuBot

## Description
...

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


## Project Structure
- **app.py**: Main file to run the project.
- **evaluation_keys.py**: File containing the keys alternative methode, used to evaluate the project.
- **evaluation_speech.py**: File containing the speech alternative methode, used to evaluate the project.
- **requirements.txt**: File containing the project dependencies.
- **/images**: Folder containing the images saved for debugging purposes.
- **/utils**: Folder containing the utility functions used in the project.
  - **/utils/models**: Folder containing the gesture recognition model used in the project (mediapipe).
  - **/utils/main_evaluation.csv**: File containing the evaluation data for the project obtained during the evaluation phase.
- **/docs**: Folder containing the project documentation.