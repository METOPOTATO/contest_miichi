from mtcnn import MTCNN
import cv2

detector = MTCNN()
video_capture = cv2.VideoCapture('meme.mp4')
while (True):
    ret, frame = video_capture.read()
    boxes = detector.detect_faces(frame)
    if boxes:
        for boxs in boxes:
            box = boxs['box']
            # conf = boxes[0]['confidence']
            x, y, w, h = box[0], box[1], box[2], box[3]
    
            # if conf > 0.8:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
 
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()