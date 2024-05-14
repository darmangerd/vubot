import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import whisper
import sounddevice as sd
import numpy as np
import queue

# Configure mediapipe
MODEL_PATH = r"./model/gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def live_audio_callback(indata, frames, time, status):
    """
    This function puts microphone data into a queue.
    """
    if status:
        print(status)
    audio_queue.put(indata.copy())


def speech_recognition(target_words: list):
    """
    This function performs the speech recognition from the microphone until any given target word is identified.
    """
    # Initialize the Whisper model
    model = whisper.load_model("base")

    # Define the sample rate and block size
    sample_rate = 16000
    block_size = 16000  # number of audio samples collected before the buffer is transcribed - 8000 processes every 0.5 s

    # Create a queue to handle real time audio data
    global audio_queue
    audio_queue = queue.Queue()

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=live_audio_callback, blocksize=block_size):
            print("Real-time transcription started. Please speak into your microphone.")
            while True:
                audio_chunk = audio_queue.get()
                if audio_chunk is not None:
                    # Convert audio chunk to numpy array
                    audio_np = np.concatenate(audio_chunk)
                    # Transcribe audio
                    result = model.transcribe(audio_np, temperature=0)
                    text = result['text'].lower()

                    # End if target words are detected
                    if any(word in text for word in target_words):
                        print(f"target word detected: {text}")
                        return text

    except KeyboardInterrupt:
        print("Exiting...")

    except Exception as e:
        print("An error occurred:", e)


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """
    This function is called when the gesture recognizer returns a result.
    It prints the index tip coordinates and checks if the point is within any of the defined shapes.

    Args:
        result: The result of the gesture recognizer.
        output_image: The output image.
        timestamp_ms: The timestamp in milliseconds.
    """
    # Check if the hand landmarks are detected
    if result.hand_landmarks:
        # Get the gesture type, we assume the first gesture detected is the relevant one
        gesture = result.gestures[0][0].category_name
        if gesture == 'Pointing_Up':
            # Get the index tip coordinates (x, y, z), 8 is the index tip landmark
            coordinates = (result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y, result.hand_landmarks[0][8].z)
            print(f'index tip coordinates: {coordinates}')
            check_point_within_shapes(coordinates)


def check_point_within_shapes(coordinates):
    """
    This function checks if the point is within any of the defined shapes.

    Args:
        coordinates: The normalized index tip coordinates (x, y, z).
    """
    # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates based on the video frame size.
    index_x = int(coordinates[0] * width)
    index_y = int(coordinates[1] * height)

    # Check if the point lies within any of the defined rectangular shapes.
    for shape_name, corners in shapes.items():
        # Check using the corners of the rectangle
        if corners[0][0] <= index_x <= corners[1][0] and corners[0][1] <= index_y <= corners[1][1]:
            print(f"Pointing inside {shape_name}")
            return
    print("Pointing outside any defined shape")


def main():

    # Define which words the user should say to trigger the object recognition
    trigger_words = ["help", "what", "whats"]
    trigger_text = speech_recognition(trigger_words)

    if trigger_text is not None: # Adapt condition to do run different actions depending on the trigger word

        # Configure the gesture recognizer options
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=print_result)

        with GestureRecognizer.create_from_options(options) as recognizer:
            # Capture video from webcam or any video source
            cap = cv2.VideoCapture(0)  # Change the parameter to the appropriate video source if necessary

            # Variables for calculating timestamp
            start_time = time.time()
            frame_count = 0

            # Retrieve the frame's width and height for coordinate conversions later
            global width, height, shapes
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            # Define shapes with names and corners at random positions
            shapes = {
                # format: [(x1, y1), (x2, y2)]
                # x1, y1: top-left corner
                # x2, y2: bottom-right corner
                "Rectangle 1": [(100, 100), (300, 300)],
                "Rectangle 2": [(350, 100), (550, 300)],
                "Rectangle 3": [(100, 350), (300, 550)],
                "Rectangle 4": [(800, 550), (1100, 750)]
            }

            while cap.isOpened():
                # Read the frame from the video capture
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame from BGR color space to RGB, as required by MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert the RGB frame to a MediaPipe Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Compute the timestamp for synchronization purposes (gesture recognition)
                current_time = time.time()
                elapsed_time = current_time - start_time
                frame_timestamp_ms = int(elapsed_time * 1000)

                # Process the frame for gesture recognition asynchronously.
                recognizer.recognize_async(mp_image, frame_timestamp_ms)

                # Draw all shapes
                for corners in shapes.values():
                    cv2.rectangle(frame, corners[0], corners[1], (255, 0, 0), 2)

                # Display the frame
                cv2.imshow('Frame', frame)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
