import cv2
import mediapipe as mp
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import whisper
import threading
import numpy as np
import sounddevice as sd
import queue

class HandGestureApp:
    """
    HandGestureApp class to run the application.
    Combines MediaPipe gesture recognition with DETR object detection.

    Args:
        model_path (str): The path to the gesture recognition model.
        debug (bool): Whether to run the application in debug mode. Default is False.
        detection_threshold (float): The object detection threshold. Default is 0.8.
    """
    def __init__(self, model_path, debug=False, detection_threshold=0.8):
        # General parameters
        self.model_path = model_path # Path to the gesture recognition model
        self.running = True # Initialize running flag
        # Gesture and Object Detection parameters
        self.index_coordinates = None # Initialize index finger coordinates
        self.width = 0 # Initialize frame width
        self.height = 0 # Initialize frame height
        self.finger_detected = False  # Initialize finger detection status
        self.debug = debug # Debug mode flag
        self.detection_threshold = detection_threshold # Object detection threshold
        self.trigger_object_detection = False  # Flag to trigger object detection
        
        # Initialize DETR model
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Initialize MediaPipe gesture recognizer
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.frame_callback
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

        # Load Whisper model
        self.whisper_model = whisper.load_model("small.en")  # Load Whisper model


    def live_audio_callback(self, indata, frames, time, status):
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
        audio_queue.put(indata.copy())


    def start_speech_recognition(self):
        """
        Start the speech recognition from the microphone until any target word is identified.
        """

        # Define the sample rate and block size
        sample_rate = 16000 # 16 kHz
        block_size = 16000  # number of audio samples collected before the buffer is transcribed - 8000 processes every 0.5 s

        # Create a queue to handle real time audio data
        global audio_queue
        audio_queue = queue.Queue()

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, callback=self.live_audio_callback, blocksize=block_size):
                print("Real-time transcription started. Please speak into your microphone.")
                while self.running:
                    audio_chunk = audio_queue.get()
                    if audio_chunk is not None:
                        # Convert audio chunk to numpy array
                        audio_np = np.concatenate(audio_chunk)
                        # Transcribe audio
                        result = self.whisper_model.transcribe(audio_np, temperature=0)

                        # format the text
                        text = result['text'].lower().translate(str.maketrans("", "", ".,!?-"))

                        if "what is this" in text :
                            self.trigger_object_detection = True
                            print("Object detection triggered.")
                            time.sleep(2)

                        else:
                            if self.debug:
                                print(f"Speech recognition: {text}")

                        
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
                print(f'Index tip coordinates: {self.index_coordinates}')
                self.finger_detected = True 
            else:
                self.finger_detected = False
        else:
            self.finger_detected = False


    def check_point_within_objects(self, frame, frame_rgb):
        """
        Check if the index finger is pointing inside any detected object.
        
        Args:
            frame (numpy.ndarray): The input frame, it is used where you are working directly 
                with OpenCV for display and drawing operations on the image.
            frame_rgb (numpy.ndarray): The RGB frame, it is used to process the image 
                with MediaPipe after conversion to ensure compatibility and accuracy of gesture recognition analyses.
            
        Returns:
            str: The result of the object detection.
        """
    
        if self.index_coordinates is None:
            return "No index finger coordinates available."
        
        index_x, index_y = self.index_coordinates

        # Process the image for object detection
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        outputs = self.model(**inputs)

        # Decode predictions from DETR model
        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        # Iterate over the detected objects
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]

            # Skip the "person" label (avoid detecting the person holding the camera)
            if label_name == "person":
                continue
            
            # Check if the object detection score is above a threshold
            if score > self.detection_threshold:
                # Convert bounding box to integer format
                box = box.int().tolist()

                # Draw bounding box (debug mode)
                if self.debug:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) 
                    score = score.item()
                    cv2.putText(frame, f"{label_name}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Check if the index finger is pointing inside the bounding box of a detected object
                if box[0] <= index_x <= box[2] and box[1] <= index_y <= box[3]:

                    # Display the object name (debug mode)
                    if self.debug:
                        # Put text with the detected object name and save a screenshot
                        cv2.putText(frame, label_name, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.imwrite(f'./image/screen_{time.time()}.png', frame)

                    # Print and return the pointed object name
                    print(f"Index finger is pointing inside the object: {label_name}")
                    return f"Index finger pointing at: {label_name}"
                
        return "Index finger pointing outside any detected object"
    

    def run(self):
        """
        Run the application.
        """

        # Capture video from webcam
        cap = cv2.VideoCapture(1) # Change to 0 for default camera
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        display_text = "Processing..."

        # Initialize the timestamp (used for gesture recognition synchronization)
        start_time = time.time()

        # Start the speech recognition thread
        speech_thread = threading.Thread(target=self.start_speech_recognition, daemon=True)
        speech_thread.start()
        print("Speech recognition thread started.")


        # Process each frame from the video capture
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame from BGR color space to RGB, as required by MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the RGB frame to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            current_time = time.time()
            elapsed_time = current_time - start_time
            frame_timestamp_ms = int(elapsed_time * 1000)

            # Process gesture recognition
            # This will trigger the frame_callback function in asynchronous mode 
            # The callback function will update the index finger coordinates and status
            self.recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # Display finger detection status
            if self.finger_detected :
                cv2.putText(frame, "Index finger detected", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 250, 30), 2)
            else:
                cv2.putText(frame, "No Index finger detected", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 250), 2)

            # Check for 'v' key press to activate object detection and potentially save a screenshot
            if cv2.waitKey(1) & 0xFF == ord('i') or self.trigger_object_detection:
                if self.debug:
                    print("riggering object detection...")
                display_text = self.check_point_within_objects(frame, frame_rgb)
                self.trigger_object_detection = False

            # Display object detection text 
            cv2.putText(frame, display_text, (int(self.width/2) - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False  # Signal to stop the threads
                break


        # Clear all resources
        cap.release()
        cv2.destroyAllWindows()
        self.recognizer.close()
        speech_thread.join()


# Run the HandGestureApp
app = HandGestureApp("./model/gesture_recognizer.task", debug=True)
app.run()