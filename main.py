import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()
clicking = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            index_x = hand_landmarks.landmark[8].x
            index_y = hand_landmarks.landmark[8].y
            thumb_x = hand_landmarks.landmark[4].x
            thumb_y = hand_landmarks.landmark[4].y

            screen_x = int(index_x * screen_w)
            screen_y = int(index_y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            dist = math.hypot(index_x - thumb_x, index_y - thumb_y)

            if dist < 0.05:
                if not clicking:
                    clicking = True
                    pyautogui.click()
            else:
                clicking = False

    cv2.imshow("Hand Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
