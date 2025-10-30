import cv2
import time
import math

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(1)


prev_cx, prev_cy = None, None
prev_time = time.time()
direction = "Center"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Draw green bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute center of face
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # Calculate time difference
        current_time = time.time()
        dt = current_time - prev_time if prev_time else 1
        prev_time = current_time

        # Compare with previous frame
        if prev_cx is not None and prev_cy is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy

            # Determine direction
            if abs(dx) > abs(dy):
                if dx > 5:
                    direction = "Right"
                elif dx < -5:
                    direction = "Left"
                else:
                    direction = "Center"
            else:
                if dy > 5:
                    direction = "Down"
                elif dy < -5:
                    direction = "Up"
                else:
                    direction = "Center"

            # Compute speed in pixels per second
            speed = math.sqrt(dx**2 + dy**2) / dt
        else:
            speed = 0.0

        # Update previous center
        prev_cx, prev_cy = cx, cy

        # Overlay text
        cv2.putText(frame, f"Direction: {direction}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed: {speed:.2f} px/s", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show video
    cv2.imshow("Face Direction Tracker", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
