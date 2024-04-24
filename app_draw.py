import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from speech.speech_recognition import real_time_transcribe

# configure mediapipe
MODEL_PATH = r"./model/gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


"""
This function is called when the gesture recognizer returns a result.
It prints the index tip coordinates and checks if the point is within any of the defined shapes.

Args:
    result: The result of the gesture recognizer.
    output_image: The output image.
    timestamp_ms: The timestamp in milliseconds.
"""
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int, text):
    # Check if the hand landmarks are detected
    if result.hand_landmarks:
        # Get the gesture type, we assume the first gesture detected is the relevant one
        gesture = result.gestures[0][0].category_name
        if gesture == 'Pointing_Up':
            # Get the index tip coordinates (x, y, z), 8 is the index tip landmark
            coordinates = (result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y, result.hand_landmarks[0][8].z)
            print(f'index tip coordinates: {coordinates}')
            check_point_within_shapes(coordinates)

"""
This function checks if the point is within any of the defined shapes.

Args:
    coordinates: The normalized index tip coordinates (x, y, z).
"""
def check_point_within_shapes(coordinates):
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
