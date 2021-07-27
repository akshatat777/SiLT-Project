import cv2
import mediapipe as mp
# import time

cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.45)
mpDraw = mp.solutions.drawing_utils

# pTime = 0
# cTime = 0
margin = 40


# to increase accuracy, you could pass in landmarks or maybe even wether its a right or left hand

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmark_listx = []
            landmark_listy = []
            for lm in handLms.landmark:
                h, w, c = img.shape

                landmark_listx.append((lm.x*w))
                landmark_listy.append((lm.y*h))
                # print(lm[1])
                # h, w, c = img.shape
                # cx, cy = int(lm.x*w), int(lm.y*h)   # the landmarks are auto-normalized by the width and height, so we have to multiply them back to scale to put them on img

                # if id == False:
                # cv2.circle(img, (cx,cy), 3, (255, 166, 48), cv2.FILLED)
                # print(handlms.landmark)
    
                # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            start = (int(max(landmark_listx))+margin, int(max(landmark_listy))+margin)
            end = (int(min(landmark_listx))-margin, int(min(landmark_listy))-margin)

            img = cv2.rectangle(img, start, end, color=(255, 166, 48), thickness=2)

            # print(start)
            # print(end)
            # print(landmark_listx)
            # print(landmark_listy)

            # print(len(landmark_list))


    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime

    # cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        # cv2.imshow("image", img)
        break

cv2.imwrite('image.jpg', img[:][end[1]:start[1], end[0]:start[0]])

# print(img[:][])


    

  
# vid.release()
cv2.destroyAllWindows()