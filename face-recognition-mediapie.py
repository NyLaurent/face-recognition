import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize face detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
    min_detection_confidence=0.5
)

# Initialize video capture
cap = cv2.VideoCapture(1)  # Use 0 for default webcam, 1 for external
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Face Detection Started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect faces
    results = face_detection.process(rgb_frame)

    # Draw face detections
    if results.detections:
        for detection in results.detections:
            # Draw bounding box and landmarks
            mp_drawing.draw_detection(frame, detection)
            
            # Get detection confidence
            confidence = detection.score[0]
            
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)
            
            # Display confidence score
            cv2.putText(frame, f'Confidence: {confidence:.2f}', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('MediaPipe Face Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
face_detection.close()
cap.release()
cv2.destroyAllWindows()
print("Face Detection Stopped.")