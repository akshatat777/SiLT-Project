import cv2
import numpy as np
import torch
from sign_recogn_joint import SignRecogJoint
from hand_cropper import crop_hand
from data_processing import normalize
from signtotext import sign_to_text
import time
import mediapipe as mp

model = SignRecogJoint()
model.load_state_dict(torch.load('sign_recogn_joint',map_location=torch.device('cpu')))
model.eval()
st = time.time()
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    texts = []
    hands = mp.solutions.hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.45)
    while True:
        def detect(texts,cap,model):
            suc, img = cap.read()
            if suc is None:
                time.sleep(0.01)
                return -1
            results = crop_hand(img,hands)
            if results is None:
                time.sleep(0.01)
                return -1
            # print(crops.shape)
            preds = model(torch.flatten(torch.tensor(results).to('cpu'),start_dim=1)).detach().numpy()
            # N, 26
            print(np.argmax(preds,axis=-1))
            max_pred = preds[np.argmax(np.max(preds,axis=1))]
            text = sign_to_text([max_pred])[0]
            texts.append(text)
            print(text)
            cv2.imshow('image',img)
            #print(normalize(crops)[0])
        if detect(texts,cap,model) == -1:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    print(texts)
