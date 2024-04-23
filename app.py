import cv2
import mediapipe as mp
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

class HandGestureApp:
    def __init__(self, model_path):
        self.model_path = model_path
        self.index_coordinates = None
        self.width = 0
        self.height = 0
        
        # Initialize DETR model
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        # Initialize MediaPipe gesture recognizer
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.print_result
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def print_result(self, result, output_image, timestamp_ms):
        if result.hand_landmarks:
            gesture = result.gestures[0][0].category_name
            if gesture == 'Pointing_Up':
                coordinates = (result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)
                self.index_coordinates = (int(coordinates[0] * self.width), int(coordinates[1] * self.height))
                print(f'Index tip coordinates: {self.index_coordinates}')

    def check_point_within_objects(self, frame):
        if self.index_coordinates is None:
            print("No index finger coordinates available.")
            return
        
        index_x, index_y = self.index_coordinates

        # Process the image for object detection
        inputs = self.processor(images=frame, return_tensors="pt")
        outputs = self.model(**inputs)

        # Decode predictions
        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.8:
                box = box.int().tolist()
                if box[0] <= index_x <= box[2] and box[1] <= index_y <= box[3]:
                    object_name = self.model.config.id2label[label.item()]
                    print(f"Index finger is pointing inside the object: {object_name}")
                    return
        print("Index finger is pointing outside any detected object")

    def run(self):
        cap = cv2.VideoCapture(0)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Initialize start time for timestamp calculation
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            # Calculate the current timestamp in milliseconds
            current_time = time.time()
            elapsed_time = current_time - start_time  # Elapsed time since the start
            frame_timestamp_ms = int(elapsed_time * 1000)  # Convert elapsed time to milliseconds


            # Process gesture recognition
            self.recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # Check for 'v' key press to activate object detection
            if cv2.waitKey(1) & 0xFF == ord('v'):
                self.check_point_within_objects(frame_rgb)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
app = HandGestureApp("./model/gesture_recognizer.task")
app.run()
