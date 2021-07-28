    import cv2
    import mediapipe

    f = cv2.cvtColor(cv2.imread('handtest.png'), cv2.COLOR_BGR2RGB)

    results = hands.process(RGBimgs)

    x_results = []
    y_results = []
    z_results = []