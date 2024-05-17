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


class VuBot:
    """
    VuBot class to run the application.
    Combines MediaPipe gesture recognition with DETR object detection.

    Args:
        model_gesture_path (str): The path to the gesture recognition model.
        debug (bool): Whether to run the application in debug mode. Default is False.
        detection_threshold (float): The object detection threshold. Default is 0.8.
    """
    def __init__(self, model_gesture_path, debug=False, detection_threshold=0.8):
        # General parameters
        self.model_gesture_path = model_gesture_path  # Path to the gesture recognition model
        self.running = True  # Initialize running flag
        self.debug = debug  # Debug mode flag
        self.width = 0  # Initialize frame width
        self.height = 0  # Initialize frame height
        self.detection_threshold = detection_threshold  # Object detection threshold

        # Gesture and Object Detection parameters
        self.index_coordinates = None  # Initialize index finger coordinates
        self.middle_finger_coordinates = None  # Initialize middle finger coordinates
        self.finger_detected = False  # Initialize finger detection status
        self.close_fist_detected = False  # Initialize closed fist detection status
        self.victory_detected = False  # Initialize victory gesture detection status
        self.trigger_object_detection = False  # Flag to trigger object detection
        self.trigger_color_detection = False  # Flag to trigger color detection
        self.last_pointed_object = None  # Last detected object name
        self.last_pointed_color = None  # Last detected object color
        self.trigger_all_objects_detection = False # Flag to trigger all objects detection

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

        # Constant sentence to trigger actions
        self.OD_POINTED_TRIGGERS = ["help object", "identify object", "object"]
        self.OD_ALL_OBJECTS_TRIGGERS = ["highlight items", "all items", "every item"]
        self.POINTED_COLOR_TRIGGER = ["help color", "identify color", "color"]
        # Countdown time for when capturing all objects
        self.COUNTDOWN_TIME = 3  

        if self.debug:
            input("Start screen recording! (press 'y' when done) ")
            # General parameters
            self.participant_id = input("Participant ID:")
            self.evaluation = {
                "ID": [],
                "version": [],
                "timelog": [],
                "task": [],
                "response": []
            }


    def start_speech_recognition(self):
        """
        Start the speech recognition from the microphone until any target word is identified.
        """

        def live_audio_callback(indata, frames, time, status):
            if status:
                print(status)
            global audio_queue
            audio_queue.put(indata.copy())

        # Initialize the Whisper model
        model = whisper.load_model("small.en")
        sample_rate = 16000  # number of audio samples collected before the buffer is transcribed - 8000 processes every 0.5 s
        block_size = 16000 * 3 

        # Create a queue to handle real time audio data
        global audio_queue
        audio_queue = queue.Queue()

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, callback=live_audio_callback, blocksize=block_size):
                print("Real-time transcription started. Please speak into your microphone.")
                while self.running:
                    audio_chunk = audio_queue.get()
                    if audio_chunk is not None:
                        # Transcribe the audio chunk
                        audio_np = np.concatenate(audio_chunk)
                        result = model.transcribe(audio_np, temperature=0)

                        # Format the transcription
                        current_transcription = result['text'].lower().strip()
                        current_transcription = ''.join(e for e in current_transcription if e.isalnum() or e.isspace())

                        if self.debug:
                            print(f"Transcription: {current_transcription}")

                        # Trigger for pointed object detection
                        if any(phrase in current_transcription for phrase in self.OD_POINTED_TRIGGERS):
                            # Lock to make sure only one thread is updating the trigger flag
                            with threading.Lock():
                                self.trigger_object_detection = True
                                print("Heard Trigger phrase for object detection...")
                                # Sleep and reset text to avoid multiple triggers
                                time.sleep(2)
                                self.current_transcription = ""

                        # Trigger for all objects detection
                        elif any(phrase in current_transcription for phrase in self.OD_ALL_OBJECTS_TRIGGERS):
                            # Lock to make sure only one thread is updating the trigger flag
                            with threading.Lock():
                                self.trigger_all_objects_detection = True
                                print("Heard Trigger phrase for all objects detection...")
                                # Sleep and reset text to avoid multiple triggers
                                time.sleep(2)
                                self.current_transcription = ""

                        # Trigger for pointed color detection
                        elif any(phrase in current_transcription for phrase in self.POINTED_COLOR_TRIGGER):
                            # Lock to make sure only one thread is updating the trigger flag
                            with threading.Lock():
                                self.trigger_color_detection = True
                                print("Heard Trigger phrase for color detection...")
                                # Sleep and reset text to avoid multiple triggers
                                time.sleep(2)
                                self.current_transcription = ""

        except Exception as e:
            print(f"Error in speech recognition: {e}")


    
    def frame_callback(self, result, output_image, timestamp_ms):
        """
        Callback function to process the result of the gesture recognizer, every time a result is received (every frame).
        It updates the status based on the detected finger.
        
        Args:
            result (mp.tasks.vision.GestureRecognizerResult): The result of the gesture recognizer.
            output_image (mp.Image): The output image.
            timestamp_ms (int): The timestamp in milliseconds.
        """

        if result.hand_landmarks:
            gesture = result.gestures[0][0].category_name

            if gesture == 'Pointing_Up':
                # Get the index tip coordinates (x, y), 8 is the index tip landmark
                coordinates = (result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)
                self.index_coordinates = (int(coordinates[0] * self.width), int(coordinates[1] * self.height))
                self.finger_detected = True 
                self.close_fist_detected = False
                self.victory_detected = False

            elif gesture == 'Closed_Fist' :
                self.close_fist_detected = True
                self.finger_detected = False
                self.victory_detected = False

            elif gesture == 'Victory':
                # TODO - victory
                index_coordinates = (result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)
                middle_finger_coordinates = (result.hand_landmarks[0][12].x, result.hand_landmarks[0][12].y)
                self.index_coordinates = (int(index_coordinates[0] * self.width), int(index_coordinates[1] * self.height))
                self.middle_finger_coordinates = (int(middle_finger_coordinates[0] * self.width), int(middle_finger_coordinates[1] * self.height))
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
            frame (numpy.ndarray): The input frame.
            frame_rgb (numpy.ndarray): The RGB frame.
            color_detection (bool): Option to indicate the color detection mode.
            
        Returns:
            str: The result of the detection of object or color.
        """
                
        if self.index_coordinates is not None:
            index_x, index_y = self.index_coordinates
        elif self.middle_finger_coordinates is not None:
            # TODO - victory
            index_x, index_y = self.middle_finger_coordinates
        else:
            return "No index finger coordinates available."

        # Process the image for object detection
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        outputs = self.model_DETR(**inputs)
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
            elif label_name == "apple":
                label_name = "ornament"
            elif label_name == "lemon":
                label_name = "lime"
            elif label_name == "banana":
                label_name = "zucchini"
            elif label_name == "scissors":
                label_name = "fork"
            elif label_name == "spoon":
                label_name = "fork"
            elif label_name == "cell phone":
                label_name = "cell phone"
            elif label_name == "book":
                label_name = "box"

            # Check if the object detection score is above a threshold
            if score > self.detection_threshold:
                box = box.int().tolist()
                
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

                    # Put text with the detected object name and save a screenshot
                    if self.debug:
                        cv2.putText(frame, label_name, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.imwrite(f'./image/screen_{time.time()}.png', frame)

        if pointed_object is not None:
            self.last_pointed_object = pointed_object

            # Return and print the name of the object if the color detection mode is off
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
            frame (numpy.ndarray): The input frame.
            frame_rgb (numpy.ndarray): The RGB frame.

        Returns:
            list: A list of dictionaries, each containing an object label and its count in the frame.
            list: A list of tuples, each containing bounding box coordinates, label, and confidence score of the detected object.
        """

        # Process the image for object detection
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        outputs = self.model_DETR(**inputs)
        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        object_counts = {} 
        bounding_boxes = [] 

        # Iterate over the detected objects
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model_DETR.config.id2label[label.item()]

            # Skip the "person" label (avoid detecting the person holding the camera)
            if label_name == "person":
                continue
            
            if score > self.detection_threshold:

                # Get score value from tensor
                score = score.item()
                box = box.int().tolist()
                bounding_boxes.append((box, label_name, score))

                if label_name in object_counts:
                    object_counts[label_name] += 1
                else:
                    object_counts[label_name] = 1

        detected_objects = [{"object": obj, "count": count} for obj, count in object_counts.items()]
        return detected_objects, bounding_boxes
    

    def sample_color_from_object(self, frame_rgb, box):
        """
        Sample the average color from the region of interest (object bounding box) in the frame.

        Args:
            frame_rgb (numpy.ndarray): The RGB frame.
            box (list): The bounding box coordinates of the object.

        Returns:
            str: The average color sampled from the object region.
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
                return "white" if l < 0.5 else "grey"
            if l < 0.2:
                return "navy"
            if 0 <= h < 30:
                return "pink"
            if 30 <= h < 90:
                return "red"
            if 90 <= h < 150:
                return "yellow"
            if 150 <= h < 210:
                return "green"
            if 210 <= h < 270:
                return "cyan"
            if 270 <= h < 330:
                return "purple"
            return "magenta"  # Wrap around for hues close to red

        def rgb_to_color_name(rgb):
            hsl = rgb_to_hsl(rgb)
            return hsl_to_color_name(hsl)

        color_name = rgb_to_color_name(rgb_value)
        return color_name


    def draw_box(self, frame, box, label, score=None):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Calculate text size and position for checking if it goes beyond the image borders
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.9
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, text_scale, thickness)

        # Position the text above the bounding box, adjusting as needed
        text_x = box[0]
        text_y = box[1] - 10 

        # Ensure the text doesn't go beyond any image borders
        if text_x < 0:
            text_x = 0

        if text_y < text_height:
            text_y = box[1] + text_height + 10

        if text_x + text_width > frame.shape[1]:
            text_x = frame.shape[1] - text_width

        if score is not None:
            cv2.putText(frame, f"{label}: {score:.2f}", (text_x, text_y), font, text_scale, (0, 255, 0), thickness)
        else:
            cv2.putText(frame, f"{label}", (text_x, text_y), font, text_scale, (0, 255, 0), thickness)

        # Start the countdown
        for i in range(self.COUNTDOWN_TIME, 0, -1):
            frame_with_boxes = frame.copy()
            text = f"{i}"
            cv2.putText(frame_with_boxes, text, (int(self.width / 2), int(self.height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 235, 255), thickness=4)
            cv2.imshow('Frame', frame_with_boxes)
            cv2.waitKey(1000)


    def run(self):
        """
        Run the application.
        """

        if self.debug:
            print(sd.query_devices())

        # Set the default input device (microphone)
        sd.default.device = 0

        # Start the speech recognition thread
        speech_thread = threading.Thread(target=self.start_speech_recognition, daemon=True)
        speech_thread.start()
        print("Speech recognition thread started.")

        # Capture video from webcam
        cap = cv2.VideoCapture(0)  # 0 for laptop, 1 for smartphone
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
            # The callback function will update the finger coordinates and status
            self.recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # Display finger detection status
            if self.victory_detected:
                cv2.putText(frame, "Victory detected", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 140, 180), thickness=3)
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
            if (self.trigger_object_detection and self.victory_detected or 
                self.trigger_object_detection and self.finger_detected or
                cv2.waitKey(1) & 0xFF == ord('o') and self.finger_detected):
                self.last_pointed_object = None
                self.last_pointed_color = None
                print("Triggering object detection...")
        
                if self.debug:
                    query_start = time.time()

                self.check_point_within_objects(frame, frame_rgb)

                if self.debug:
                    query_end = time.time()
                    query_duration = query_end - query_start
                    self.evaluation["ID"].append(self.participant_id)
                    self.evaluation["version"].append('speech')  # Different for alternate version to evaluate!
                    self.evaluation["timelog"].append(query_duration)
                    self.evaluation["task"].append('object')
                    self.evaluation["response"].append(self.last_pointed_object)


            # Check if the condition for launching the color detection are met
            elif (self.trigger_color_detection and self.victory_detected or
                  self.trigger_color_detection and self.finger_detected or
                  cv2.waitKey(1) & 0xFF == ord('c') and self.finger_detected):
                self.last_pointed_object = None
                self.last_pointed_color = None
                print("Triggering color detection...")

                if self.debug:
                    query_start = time.time()

                self.check_point_within_objects(frame, frame_rgb, color_detection=True)

                if self.debug:
                    query_end = time.time()
                    query_duration = query_end - query_start
                    self.evaluation["ID"].append(self.participant_id)
                    self.evaluation["version"].append('speech')  # Different for alternate version to evaluate!
                    self.evaluation["timelog"].append(query_duration)
                    self.evaluation["task"].append('color')
                    self.evaluation["response"].append(self.last_pointed_color)


            # Check if the condition for launching the detection of all objects are met
            elif self.trigger_all_objects_detection and self.close_fist_detected:
                self.last_pointed_object = None
                self.last_pointed_color = None
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

            # Reset the trigger flags if they were activated (to avoid multiple triggers)
            self.trigger_all_objects_detection = False
            self.trigger_object_detection = False
            self.trigger_color_detection = False


            # Check for 'q' key press to exit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):

                if self.debug:
                    print("Saving evaluation data...")
                    # Save data for evaluation before leaving
                    df_evaluation = pd.DataFrame.from_dict(self.evaluation)
                    df_evaluation.to_csv(f"{self.participant_id}_{self.evaluation['version'][0]}.csv")  # save it as separate file
                    df_evaluation.to_csv('./utils/main_evaluation.csv', mode='a', index=True, header=False) # append data to the main file

                self.running = False  # Signal to stop the threads
                break


        print("Exiting the application...")
        # Clear all resources
        cap.release()
        cv2.destroyAllWindows()
        self.recognizer.close()
        speech_thread.join()


# Run VuBot application
app = VuBot("./utils/model/gesture_recognizer.task", debug=True)
app.run()