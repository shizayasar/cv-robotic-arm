import cv2
import mediapipe as mp

# Set up MediaPipe hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Open the webcam (0 = built-in camera)
cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)                       # mirror the image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # MediaPipe expects RGB
    result = hands.process(rgb)

    # Draw the 21 hand landmarks if a hand is found
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):            # press q to quit
        break

cap.release()
cv2.destroyAllWindows()