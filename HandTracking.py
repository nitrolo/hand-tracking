import cv2
import mediapipe as mp

# Start capturing webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not capture.")

# Display the video stream
while cap.isOpened():
    res, frame = cap.read()
    if res:
        # Convert color channels of the frame to RGB for mediapipe
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the frame
        cv2.imshow(frame_RGB)
        # Press esc to exit
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break

# Close the webcam feed
cap.release()
cv2.destroyAllWindows()
