import cv2
import numpy as np
import torch
from sign_recog_cnn import SignRecogCNN
from hand_cropper import crop_hand
from data_processing import normalize
from signtotext import sign_to_text
import time

model = SignRecogCNN()
model.load_state_dict(torch.load('sign_recogn_cnn_larger',map_location=torch.device('cpu')))
model.eval()
st = time.time()
with torch.no_grad():
    cap = cv2.VideoCapture(0)
    texts = []
    while True:
        def detect(texts,cap,model):
            crops = np.array(crop_hand(cap))
            if crops.shape[0] == 0:
                time.sleep(0.01)
                return -1
            # print(crops.shape)
            preds = model(torch.tensor(normalize(crops)).to('cpu')).detach().numpy()
            # N, 26
            print(np.argmax(preds,axis=-1))
            max_pred = preds[np.argmax(np.max(preds,axis=1))]
            cv2.imshow('crop',crops[0])
            text = sign_to_text([max_pred])[0]
            texts.append(text)
            print(text)
            #print(normalize(crops)[0])
        if detect(texts,cap,model) == -1:
            continue
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            break
    print(texts)
