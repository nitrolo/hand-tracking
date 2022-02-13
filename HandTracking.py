import cv2
import mediapipe as mp

# Start capturing webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not capture.")

# Creating an object of mediapipe's Hands class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
# Using drawing_utils to allow us to draw the hand skeleton
mp_draw = mp.solutions.drawing_utils

# Display the video stream
while cap.isOpened():
    res, frame = cap.read()
    if res:
        # Convert color channels of the frame to RGB for mediapipe
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the current frame using mediapipe's Hand class
        results = hands.process(frame_RGB)

        # Check if hand(s) is/are detected and display landmark points on each hand
        if results.multi_hand_landmarks:
            for single_hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and their connections on each hand
                mp_draw.draw_landmarks(
                    frame, single_hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
