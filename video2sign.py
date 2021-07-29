import cv2
import numpy as np
import torch
from sign_recogn_joint import SignRecogJoint
from hand_cropper import crop_hand
from data_processing import normalize_joints
import time
import mediapipe as mp
from signtotext import sign_to_text

model = SignRecogJoint()
model.load_state_dict(torch.load('sign_recogn_joint',map_location=torch.device('cpu')))
model.eval()
st = time.time()
alphabet = 'abcdefghijklmnopqrstuvwxyz'
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    texts = []
    pred_scores = []
    hands = mp.solutions.hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.45)
    while True:
        def detect(texts,pred_scores,cap,model):
            suc, img = cap.read()
            if suc is None:
                time.sleep(0.01)
                return -1
            results = crop_hand(img,hands)
            if results is None:
                texts.append(' ')
                pred_scores.append(0)
                time.sleep(0.01)
                return -1
            # print(crops.shape)
            results = normalize_joints(results)
            preds = model(torch.flatten(torch.tensor(results).to('cpu'),start_dim=1)).detach().numpy()[0]
            # N, 26
            pred_scores.append(np.max(preds))
            text = alphabet[np.argmax(preds)]
            texts.append(text)
            print(text)
            cv2.imshow('image',img)
            #print(normalize(crops)[0])
        if detect(texts,pred_scores,cap,model) == -1:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    print(texts)
    pred_scores = np.stack(pred_scores,axis=0)
    print(pred_scores)
    print(pred_scores.shape)
    print(sign_to_text(texts, pred_scores))