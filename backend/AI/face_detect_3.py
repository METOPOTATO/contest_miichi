import cv2 


modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


# blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)

# net.setInput(blob)


video_capture = cv2.VideoCapture('meme.mp4')
while (True):
    
    ret, frame = video_capture.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.8:

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            cv2.rectangle(frame, (x1, y1), (x2 , y2), (255, 255, 255), 1)
    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()