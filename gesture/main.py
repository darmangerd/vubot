# prep
# download mediapipe from terminal
# pip install -q mediapipe

# download model from terminal
# wget -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

# running interference and visualising the results
# run the gesture recognition using MediaPipe ON IMAGES

# STEP 1: Imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

# STEP 3: Create a gesture recognizer instance with the live stream mode:
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# STEP 2: Model
model_path = r"C:\Users\olivi\switchdrive2\Institution\_DIGITAL_NEUROSCIENCE\Multimodal User Interfaces\MMUI_Project\gesture_recognizer.task"


# def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
#     print('gesture recognition result: {}'.format(result))


# OPTION: prints landmarks and gesture type only when hand is in frame
# def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
#     if result.hand_landmarks:
#         print(result.gestures)
#         print(f'index tip coordinates: {result.hand_landmarks[0][7]}')


# def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
#     if result.hand_landmarks:
#         gesture = result.gestures[0][0].category_name
#         print(f'gesture: {gesture}')
#         coordinates = (result.hand_landmarks[0][7].x, result.hand_landmarks[0][7].y, result.hand_landmarks[0][7].z)
#         print(f'index tip coordinates: {coordinates}')


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        gesture = result.gestures[0][0].category_name
        if gesture == 'Pointing_Up':
            coordinates = (result.hand_landmarks[0][7].x, result.hand_landmarks[0][7].y, result.hand_landmarks[0][7].z)
            print(f'index tip coordinates: {coordinates}')


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:

    # Capture video from webcam or any video source
    cap = cv2.VideoCapture(0)  # Change the parameter to the appropriate video source if necessary

    # Variables for calculating timestamp
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Calculate the timestamp in milliseconds
        current_time = time.time()
        elapsed_time = current_time - start_time
        frame_timestamp_ms = int(elapsed_time * 1000)

        # Send live image data to perform gesture recognition.
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

