import cv2
import mediapipe as mp
import time
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

class HandGestureApp:
    def __init__(self, model_path, debug=False):
        self.model_path = model_path
        self.index_coordinates = None
        self.width = 0
        self.height = 0
        self.finger_detected = "No index finger detected"  # Initial text when no finger is detected
        self.debug = debug
        
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
                self.finger_detected = "Index finger detected"  # Update status when index finger is detected
            else:
                self.finger_detected = "No index finger detected"
        else:
            self.finger_detected = "No index finger detected"  # Maintain status when no finger is detected

    def check_point_within_objects(self, frame, frame_rgb):
        if self.index_coordinates is None:
            return "No index finger coordinates available."
        
        index_x, index_y = self.index_coordinates

        # Process the image for object detection
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        outputs = self.model(**inputs)

        # Decode predictions
        target_sizes = torch.tensor([frame.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]


        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()]

            # Skip the "person" label
            if label_name == "person":
                continue
            
            if score > 0.8:
                box = box.int().tolist()

                # Draw bounding box (debug mode)
                if self.debug:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2) 
                    score = score.item()
                    cv2.putText(frame, f"{label_name}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check if the index finger is pointing inside the bounding box of a detected object
                if box[0] <= index_x <= box[2] and box[1] <= index_y <= box[3]:

                    # Display the object name (debug mode)
                    if self.debug:
                        # Put text with the detected object name
                        cv2.putText(frame, label_name, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Save a screenshot when the index finger is pointing inside the object
                        cv2.imwrite(f'./image/screen_{time.time()}.png', frame)

                    # Print and return the object name
                    print(f"Index finger is pointing inside the object: {label_name}")
                    return f"Index finger pointing at: {label_name}"
        return "Index finger pointing outside any detected object"

    def run(self):
        cap = cv2.VideoCapture(0)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        display_text = "Processing..."

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            # Calculate the current timestamp in milliseconds
            current_time = time.time()
            elapsed_time = current_time - start_time
            frame_timestamp_ms = int(elapsed_time * 1000)

            # Process gesture recognition
            self.recognizer.recognize_async(mp_image, frame_timestamp_ms)

            # Display finger detection status
            cv2.putText(frame, self.finger_detected, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Check for 'v' key press to activate object detection and potentially save a screenshot
            if cv2.waitKey(1) & 0xFF == ord('v'):
                display_text = self.check_point_within_objects(frame, frame_rgb)

            # Display object detection text
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Usage
app = HandGestureApp("./model/gesture_recognizer.task", debug=True)
# app.run()
