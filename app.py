import cv2
import mediapipe as mp
import time

import pandas as pd
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import whisper
import threading
import numpy as np
import sounddevice as sd
import queue
import colorsys


class HandGestureApp:
    """
    HandGestureApp class to run the application.
    Combines MediaPipe gesture recognition with DETR object detection.

    Args:
        model_gesture_path (str): The path to the gesture recognition model.
        debug (bool): Whether to run the application in debug mode. Default is False.
        detection_threshold (float): The object detection threshold. Default is 0.8.
    """
    def __init__(self, model_gesture_path, debug=False, detection_threshold=0.5, display_boxes=False):
        # General parameters
        self.participant_id = input("Participant ID:")
        self.evaluation = {
            "ID": [],
            "version": [],
            "timelog": [],
            "task": [],
            "response": []
        }
        self.model_gesture_path = model_gesture_path  # Path to the gesture recognition model
        self.running = True  # Initialize running flag
        # Gesture and Object Detection parameters
        self.index_coordinates = None  # Initialize index finger coordinates
        self.middle_finger_coordinates = None  # Initialize middle finger coordinates
        self.width = 0  # Initialize frame width
        self.height = 0  # Initialize frame height
        self.finger_detected = False  # Initialize finger detection status
        self.close_fist_detected = False  # Initialize closed fist detection status
        self.victory_detected = False  # Initialize victory gesture detection status
        self.debug = debug  # Debug mode flag
        self.detection_threshold = detection_threshold  # Object detection threshold
        self.trigger_object_detection = False  # Flag to trigger object detection
        self.trigger_color_detection = False  # Flag to trigger color detection
        self.last_pointed_object = None  # Last detected object name
        self.last_pointed_color = None  # Last detected object color
        self.trigger_all_objects_detection = False # Flag to trigger all objects detection
        self.previous_transcription = ""  # Variable to store the previous transcription
        self.detection_box = None  # Variable to store the square boxes around detected objects
        self.score = None  # Variable to store the detection confidence score
        self.display_boxes = display_boxes  # Flag for displaying the square boxes around detected objects

        # Initialize DETR model
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model_DETR = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Initialize MediaPipe gesture recognizer
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_gesture_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.frame_callback
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

        # Load Whisper model
        self.whisper_model = whisper.load_model("small.en")  # Load Whisper model

        # Constant sentence to trigger actions
        # self.OD_POINTED_TRIGGERS = ["what is this", "help me what is this", "identify this"]
        # self.OD_ALL_OBJECTS_TRIGGERS = ["capture all objects", "capture objects"]
        # self.POINTED_COLOR_TRIGGER = "what color is this", "what color is that"
        # self.QUESTION_TRIGGER = "help is it", "help is this", "help me is that"
        # self.QUESTION_COLOR_TRIGGER = "what color is this", "what is the color of this", "what color is that"
        # self.COUNTDOWN_TIME = 5  # Countdown time for capturing all objects

        self.OD_POINTED_TRIGGERS = ["help object", "object"]
        self.OD_ALL_OBJECTS_TRIGGERS = ["help highlight objects", "all objects", "all", "everything"]
        self.POINTED_COLOR_TRIGGER = ["help color", "help colour", "color"]
        self.QUESTION_TRIGGER = "help"
        self.QUESTION_COLOR_TRIGGER = "help"
        self.COUNTDOWN_TIME = 5  # Countdown time for capturing all objects

    def start_speech_recognition(self):
        """
        Start the speech recognition from the microphone until any target word is identified.
        """


        def live_audio_callback(indata, frames, time, status):
            """
            Live audio callback function to put microphone data into a queue.

            Args:
                indata (numpy.ndarray): The input audio data.
                frames (int): The number of frames.
                time (sounddevice.CallbackTimeInfo): The time information.
                status (sounddevice.CallbackFlags): The callback flags.
            """
            if status:
                print(status)
            global audio_queue
            audio_queue.put(indata.copy())

        # Initialize the Whisper model
        model = whisper.load_model("small.en")

        # Define the sample rate and block size
        sample_rate = 16000 # don't change this
        block_size = 16000 * 3  # number of audio samples collected before the buffer is transcribed - 8000 processes every 0.5 s

        # Create a queue to handle real time audio data
        global audio_queue
        audio_queue = queue.Queue()

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, callback=live_audio_callback, blocksize=block_size):
                print("Real-time transcription started. Please speak into your microphone.")
                while self.running:
                    audio_chunk = audio_queue.get()
                    if audio_chunk is not None:
                        # Convert audio chunk to numpy array
                        audio_np = np.concatenate(audio_chunk)
                        # Transcribe audio
                        result = model.transcribe(audio_np, temperature=0)

                        # Format the transcription
                        current_transcription = result['text'].lower().strip()
                        current_transcription = ''.join(e for e in current_transcription if e.isalnum() or e.isspace())

                        if self.debug:
                            print(f"Transcription: {current_transcription}")

                        # Combine the previous and current transcriptions for better handling user speech
                        combined_transcription = f"{self.previous_transcription} {current_transcription}".strip()

                        if any(phrase in combined_transcription for phrase in self.OD_POINTED_TRIGGERS):
                            # Lock to make sure only one thread is updating the trigger flag
                            with threading.Lock():
                                self.trigger_object_detection = True
                                print("Heard Trigger phrase for object detection...")
                                # Sleep and reset text to avoid multiple triggers
                                time.sleep(2)
                                self.current_transcription = ""

                        elif any(phrase in combined_transcription for phrase in self.OD_ALL_OBJECTS_TRIGGERS):
                            # lock to make sure only one thread is updating the trigger flag
                            with threading.Lock():
                                self.trigger_all_objects_detection = True
                                print("Heard Trigger phrase for all objects detection...")
                                # Sleep and reset text to avoid multiple triggers
                                time.sleep(2)
                                self.current_transcription = ""

                        elif any(phrase in combined_transcription for phrase in self.POINTED_COLOR_TRIGGER):
                            # lock to make sure only one thread is updating the trigger flag
                            with threading.Lock():
                                self.trigger_color_detection = True
                                print("Heard Trigger phrase for color detection...")
                                # Sleep and reset text to avoid multiple triggers
                                time.sleep(2)
                                self.current_transcription = ""

                        # Update the previous transcription
                        self.previous_transcription = current_transcription
                        

        except Exception as e:
            print(f"Error in speech recognition: {e}")


    
    def frame_callback(self, result, output_image, timestamp_ms):
        """
        Callback function to process the result of the gesture recognizer, every time a result is received (every frame).
        It prints the index tip coordinates and updates the status based on the detected finger. 
        It also checks if the index finger is pointing inside any detected object.
        
        Args:
            result (mp.tasks.vision.GestureRecognizerResult): The result of the gesture recognizer.
            output_image (mp.Image): The output image.
            timestamp_ms (int): The timestamp in milliseconds.
            
        Returns:    
            str: The result of the object detection.        
        """

        # Check if the hand landmarks are detected
        if result.hand_landmarks:
            # Get the gesture type, we assume the first gesture detected is the relevant one
            gesture = result.gestures[0][0].category_name

            if gesture == 'Pointing_Up':
                # Get the index tip coordinates (x, y), 8 is the index tip landmark
                coordinates = (result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)
                self.index_coordinates = (int(coordinates[0] * self.width), int(coordinates[1] * self.height))
                # print(f'Index tip coordinates: {self.index_coordinates}')
                self.finger_detected = True 
                self.close_fist_detected = False
                self.victory_detected = False

            # Check if the gesture is a closed fist to trigger all objects detection
            # we use the trigger_all_objects flag to avoid multiple triggers (it is reset when the gesture changes)
            elif gesture == 'Closed_Fist' :
                self.close_fist_detected = True
                self.finger_detected = False
                self.victory_detected = False

            # Check for victory sign gesture and get index and middle fingertip coordinates
            elif gesture == 'Victory':
                index_coordinates = (result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)
                middle_finger_coordinates = (result.hand_landmarks[0][12].x, result.hand_landmarks[0][12].y)
                self.index_coordinates = (int(index_coordinates[0] * self.width), int(index_coordinates[1] * self.height))
                # print(f'Index tip coordinates: {self.index_coordinates}')
                self.middle_finger_coordinates = (int(middle_finger_coordinates[0] * self.width), int(middle_finger_coordinates[1] * self.height))
                # print(f'Middle finger tip coordinates: {self.middle_finger_coordinates}')
                self.victory_detected = True
                self.close_fist_detected = False
                self.finger_detected = False

            else:
                self.finger_detected = False
                self.close_fist_detected = False
                self.victory_detected = False

        else:
            self.finger_detected = False
            self.close_fist_detected = False
            self.victory_detected = False



    def check_point_within_objects(self, frame, frame_rgb, color_detection=False):
        """
        Check if the index finger is pointing inside any detected object.
        
        Args:
            frame (numpy.ndarray): The input frame, it is used where you are working directly 
                with OpenCV for display and drawing operations on the image.
            frame_rgb (numpy.ndarray): The RGB frame, it is used to process the image 
                with MediaPipe after conversion to ensure compatibility and accuracy of gesture recognition analyses.
            color_detection (bool): Option to indicate the color detection mode.
            
        Returns:
            str: The result of the detection of object or color.
        """
    
        if self.index_coordinates is None:
            return "No index finger coordinates available."
        
        index_x, index_y = self.index_coordinates

        # Process the image for object detection
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        outputs = self.model_DETR(**inputs)

        # Decode predictions from DETR model
        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        pointed_object = None
        pointed_color = None

        # Iterate over the detected objects
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model_DETR.config.id2label[label.item()]

            # Skip the "person" label (avoid detecting the person holding the camera)
            if label_name == "person":
                continue
            
            # Check if the object detection score is above a threshold
            if score > self.detection_threshold:
                # Convert bounding box to integer format
                box = box.int().tolist()
                # Update score
                # self.score = score.item()

                # Draw bounding box (debug mode)
                if self.debug:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    score = score.item()
                    cv2.putText(frame, f"{label_name}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Check if the index finger is pointing inside the bounding box of a detected object
                if box[0] <= index_x <= box[2] and box[1] <= index_y <= box[3]:
                    pointed_object = label_name

                    # Sample the average color from the object
                    if color_detection:
                        pointed_color = self.sample_color_from_object(frame_rgb, box)

                    # Display the object name (debug mode)
                    if self.debug:
                        # Put text with the detected object name and save a screenshot
                        cv2.putText(frame, label_name, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.imwrite(f'./image/screen_{time.time()}.png', frame)

        if pointed_object is not None:
            self.last_pointed_object = pointed_object
            self.detection_box = box
            self.score = score

            # Return adn print the detected object in object detection mode
            if not color_detection:
                print(f"Selected object: {pointed_object}")
                return pointed_object

            # Return and print the color of the object if the color detection mode is on
            else:
                self.last_pointed_color = pointed_color
                print(f"Color of the selected object: {pointed_color}")
                return pointed_color

        else:
            print("Index finger pointing outside any detected object")
            return None


    def capture_all_objects(self, frame, frame_rgb):
        """
        Capture all detected objects in the frame.
        
        Args:
            frame (numpy.ndarray): The input frame, it is used where you are working directly 
                with OpenCV for display and drawing operations on the image.
            frame_rgb (numpy.ndarray): The RGB frame, it is used to process the image 
                with MediaPipe after conversion to ensure compatibility and accuracy of gesture recognition analyses.
        """

        # Process the image for object detection
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        outputs = self.model_DETR(**inputs)

        # Decode predictions from DETR model
        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        object_counts = {} # Initialize object counts
        bounding_boxes = [] # Initialize bounding boxes

        # Iterate over the detected objects
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model_DETR.config.id2label[label.item()]

            # Skip the "person" label (avoid detecting the person holding the camera)
            if label_name == "person":
                continue
            
            # Check if the object detection score is above a threshold
            if score > self.detection_threshold:

                # Update score
                # self.score = score.item()

                # Get score value from tensor
                score = score.item()

                # Convert bounding
                box = box.int().tolist()

                # save bounding box and label
                bounding_boxes.append((box, label_name, score))

                # Update the object counts
                if label_name in object_counts:
                    object_counts[label_name] += 1
                else:
                    object_counts[label_name] = 1


        # Convert the dictionary to a list of dictionaries
        detected_objects = [{"object": obj, "count": count} for obj, count in object_counts.items()]

        return detected_objects, bounding_boxes

    def sample_color_from_object(self, frame_rgb, box):
        """
        Sample the average color from the region of interest (object bounding box) in the frame.

        Args:
            frame_rgb (numpy.ndarray): The RGB frame.
            box (list): The bounding box coordinates of the object (in format [x_min, y_min, x_max, y_max]).

        Returns:
            tuple: The average color sampled from the object region (in BGR format).
        """
        x_min, y_min, x_max, y_max = box

        # Extract the region of interest (object) from the frame
        object_region = frame_rgb[y_min:y_max, x_min:x_max]

        # Calculate the average color of the object region
        avg_color = np.mean(object_region, axis=(0, 1))

        rgb_value = tuple(int(round(x)) for x in avg_color)

        def rgb_to_hsl(rgb):
            r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin

            # Hue calculation
            if delta == 0:
                h = 0
            elif cmax == r:
                h = 60 * (((g - b) / delta) % 6)
            elif cmax == g:
                h = 60 * (((b - r) / delta) + 2)
            else:
                h = 60 * (((r - g) / delta) + 4)

            # Lightness calculation
            l = (cmax + cmin) / 2

            # Saturation calculation
            if delta == 0:
                s = 0
            else:
                s = delta / (1 - abs(2 * l - 1))

            return h, s, l

        def hsl_to_color_name(hsl):
            h, s, l = hsl
            if s < 0.1:
                return "gray" if l < 0.5 else "white"
            if l < 0.2:
                return "black"
            if 0 <= h < 30:
                return "red"
            if 30 <= h < 90:
                return "yellow"
            if 90 <= h < 150:
                return "green"
            if 150 <= h < 210:
                return "cyan"
            if 210 <= h < 270:
                return "blue"
            if 270 <= h < 330:
                return "magenta"
            return "red"  # Wrap around for hues close to red

        def rgb_to_color_name(rgb):
            hsl = rgb_to_hsl(rgb)
            return hsl_to_color_name(hsl)

        color_name = rgb_to_color_name(rgb_value)
        return color_name


    def draw_box(self, frame, box, label, score=None):
        # print('box', box)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Calculate text size and position for checking if it goes beyond the image borders
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.9
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, text_scale, thickness)

        # Position the text above the bounding box, adjusting as needed
        text_x = box[0]
        text_y = box[1] - 10  # default position above the bounding box

        # Ensure the text doesn't go beyond any image borders
        if text_x < 0:
            text_x = 0

        if text_y < text_height:
            text_y = box[1] + text_height + 10

        if text_x + text_width > frame.shape[1]:
            text_x = frame.shape[1] - text_width

        # Draw the label text; only include score if it is set to a value
        if score is not None:
            cv2.putText(frame, f"{label}: {score:.2f}", (text_x, text_y), font, text_scale, (0, 255, 0), thickness)
        else:
            cv2.putText(frame, f"{label}", (text_x, text_y), font, text_scale, (0, 255, 0), thickness)

        # Start the countdown
        for i in range(self.COUNTDOWN_TIME, 0, -1):
            # Create a copy of the frame with bounding boxes drawn
            frame_with_boxes = frame.copy()

            # Draw the countdown text
            text = f"{i}"
            cv2.putText(frame_with_boxes, text, (int(self.width / 2), int(self.height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 235, 255), thickness=4)

            cv2.imshow('Frame', frame_with_boxes)

            # Wait for 1 second
            cv2.waitKey(1000)


    def run(self):
        """
        Run the application.
        """

        if self.debug:
            # print sounddevice info
            print(sd.query_devices())

        # Set the default input device (microphone)
        sd.default.device = 0

        # Start the speech recognition thread
        speech_thread = threading.Thread(target=self.start_speech_recognition, daemon=True)
        speech_thread.start()
        print("Speech recognition thread started.")

        # Capture video from webcam
        # TODO: change video input
        cap = cv2.VideoCapture(0)  # 0 for laptop, 1 for smartphone
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.width = 1920
        # self.height = 1080

        # Initialize the timestamp (used for gesture recognition synchronization)
        start_time = time.time()

        print("Starting the application...")

        # Process each frame from the video capture
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame from BGR color space to RGB, as required by MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the RGB frame to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Calculate the elapsed time in milliseconds (used for gesture recognition synchronization)
            current_time = time.time()
            elapsed_time = current_time - start_time
            frame_timestamp_ms = int(elapsed_time * 1000)

            # Process gesture recognition
            # This will trigger the frame_callback function in asynchronous mode 
            # The callback function will update the index finger coordinates and status
            self.recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # Display finger detection status
            if self.victory_detected:
                cv2.putText(frame, "Victory detected", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 50, 50), thickness=3)
            elif self.finger_detected:
                cv2.putText(frame, "Index finger detected", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 30), thickness=3)
            elif self.close_fist_detected:
                cv2.putText(frame, "Closed fist detected", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 50, 50), thickness=3)
            else:
                cv2.putText(frame, "No Gesture detected", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 250), thickness=3)


            # Display on the screen the last pointed object or its color
            if self.last_pointed_object is not None and self.last_pointed_color is None:
                cv2.putText(frame, f"Last pointed object: {self.last_pointed_object}", (int(self.width / 2) - 150, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), thickness=4)
            elif self.last_pointed_color is not None:
                cv2.putText(frame, f"Color of the last pointed object: {self.last_pointed_color}",
                            (int(self.width / 2) - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),
                            thickness=4)
            else:
                cv2.putText(frame, "Did not point at any object", (int(self.width/2) - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), thickness=3)


            # Check if the condition for launching the single object detection are met
            if self.trigger_object_detection and self.victory_detected or self.trigger_object_detection and self.finger_detected:
                self.last_pointed_object = None
                self.last_pointed_color = None
                # self.score = None
                print("Triggering object detection...")
                query_start = time.time()
                self.check_point_within_objects(frame, frame_rgb)
                query_end = time.time()
                query_duration = query_end - query_start
                self.evaluation["ID"].append(self.participant_id)
                self.evaluation["version"].append('speech')  # Different for alternate version to evaluate!
                self.evaluation["timelog"].append(query_duration)
                self.evaluation["task"].append('object')
                self.evaluation["response"].append(self.last_pointed_object)


                # Draw bounding boxes and labels on the frame
                if self.display_boxes and self.detection_box is not None:
                    box = self.detection_box
                    label = self.last_pointed_object
                    # score = self.score
                    self.draw_box(frame, box, label)

            # Check if the condition for launching the color detection are met
            elif self.trigger_color_detection and self.victory_detected or self.trigger_color_detection and self.finger_detected:
                self.last_pointed_object = None
                self.last_pointed_color = None
                # self.score = None
                print("Triggering color detection...")
                query_start = time.time()
                self.check_point_within_objects(frame, frame_rgb, color_detection=True)
                query_end = time.time()
                query_duration = query_end - query_start
                self.evaluation["ID"].append(self.participant_id)
                self.evaluation["version"].append('speech')  # Different for alternate version to evaluate!
                self.evaluation["timelog"].append(query_duration)
                self.evaluation["task"].append('color')
                self.evaluation["response"].append(self.last_pointed_color)

                # Draw bounding boxes and labels on the frame
                if self.display_boxes and self.detection_box is not None:
                    box = self.detection_box
                    label = self.last_pointed_color
                    # score = self.score
                    self.draw_box(frame, box, label)

            # Check if the condition for launching the detection of all objects are met
            elif self.trigger_all_objects_detection and self.close_fist_detected:
                self.last_pointed_object = None
                self.last_pointed_color = None
                self.score = None
                print("Triggering all objects detection...")

                # Draw bounding boxes and labels on the frame if objects are detected
                detected_objects, bounding_boxes = self.capture_all_objects(frame, frame_rgb)
                if detected_objects is None:
                    print("No objects detected.")
                    continue
                print("Detected objects:")
                for obj in detected_objects:
                    print(f"{obj['object']}: {obj['count']}")
                for box, label, score in bounding_boxes:
                    self.draw_box(frame, box, label, score)


            # Display the frame 
            cv2.imshow('Frame', frame)

            # Reset the trigger flags if they were activated
            self.trigger_all_objects_detection = False
            self.trigger_object_detection = False
            self.trigger_color_detection = False


            # Check for 'q' key press to exit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                df_evaluation = pd.DataFrame.from_dict(self.evaluation)
                print(
                    f"{df_evaluation = }"
                )
                # df_evaluation.to_excel(f"/Users/sophiecaroni/vubot/{self.participant_id}_{self.evaluation['version']}.xlsx")
                df_evaluation.to_csv(f"{self.participant_id}_{self.evaluation['version'][0]}.csv")

                self.running = False  # Signal to stop the threads
                break


        print("Exiting the application...")
        # Clear all resources
        cap.release()
        cv2.destroyAllWindows()
        self.recognizer.close()
        speech_thread.join()


# Run the HandGestureApp
app = HandGestureApp("./model/gesture_recognizer.task", debug=False, display_boxes=False)
app.run()