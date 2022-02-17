import cv2
import mediapipe as mp
import time

# Start capturing webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not capture.")

# Creating an object of mediapipe's Hands class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Using drawing_utils to allow us to draw the hand skeleton
mp_draw = mp.solutions.drawing_utils

prev_frame_time, curr_frame_time = 0, 0

# Display the video stream
while cap.isOpened():
    res, frame = cap.read()
    # Flip the frame to display like a mirror
    frame = cv2.flip(frame, 3)
    if res:
        # Convert color channels of the frame to RGB for mediapipe
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the current frame using mediapipe's Hand class
        results = hands.process(frame_RGB)

        # Check if hand(s) is/are detected and display landmark points on each hand
        if results.multi_hand_landmarks:
            for single_hand_landmarks in results.multi_hand_landmarks:
                # Get position of each landmark
                for landmark_id, landmark in enumerate(single_hand_landmarks.landmark):
                    height, width, channels = frame.shape
                    # Convert ratio values of the landmark coordinates to pixels
                    landmark_x, landmark_y = int(landmark.x * width), int(landmark.y * height)
                    # Detect the tip of each finger
                    if landmark_id != 0 and landmark_id % 4 == 0:
                        cv2.circle(frame, (landmark_x, landmark_y), 15,
                                   (0, 255, 0), cv2.FILLED)
                # Draw landmarks and their connections on each hand
                mp_draw.draw_landmarks(
                    frame, single_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate the FPS
        curr_frame_time = time.time()
        fps = 1 / (curr_frame_time - prev_frame_time)
        prev_frame_time = curr_frame_time

        # Display FPS on the frame
        cv2.putText(frame, str(int(fps)), (20, 60), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 200), 3)

        # Display the frame
        cv2.imshow("Video", frame)
        # Press esc to exit
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break

# Close the webcam feed
cap.release()
cv2.destroyAllWindows()
