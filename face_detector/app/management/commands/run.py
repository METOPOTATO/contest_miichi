from tkinter.constants import E, LEFT, RIGHT
from PIL import Image, ImageTk
import numpy as np
from tkinter import ttk
from tkinter import Tk,Frame, Button,Label,Entry,Text,StringVar, W, TOP
import cv2
import os
import pyttsx3
# cap = cv2.VideoCapture(0)
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from django.core.management.base import BaseCommand
from app.models import People
# from models import People
# face_detector/app/management/commands/bvlc_googlenet.caffemodel
modelFile = "app/management/commands/lib/opencv_face_detector_uint8.pb"
configFile = "app/management/commands/lib/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

folder = 'app/management/commands/images'

face_model = load_model('app/management/commands/lib/facenet_keras.h5')


def read_image_from_files(): # load_data
    # client = MongoClient('mongodb://localhost:27017')
    # db = client.contest_michi
    print('Reload data')
    embeddings = []
    labels = []
    for person_files in  os.listdir(folder):
        print(person_files)
        for file_name in  os.listdir(folder +'/'+person_files):
            img_dir = os.path.join(folder +'/'+person_files, file_name)
            print
            img = cv2.imread(img_dir)
            img = get_face_from_img(img)
            # cv2.waitKey(1)
            if img is not None:
                # cv2.imshow('img', img)
                embedding = get_embedding(img)

                embeddings.append(embedding)
                labels.append(person_files)

    labels = np.array(labels)
    embeddings = np.array(embeddings)

    in_encoder = Normalizer(norm='l2')
    out_encoder = LabelEncoder()
    out_encoder.fit(labels)

    X_transformed = in_encoder.transform(embeddings)
    y_transformed = out_encoder.transform(labels)
    svc_model = SVC(kernel='linear', probability=True)
    svc_model.fit(X_transformed, y_transformed)

    return svc_model, in_encoder, out_encoder
    

def get_face_from_img(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            cv2.rectangle(img, (x1, y1), (x2 , y2), (255, 255, 255), 1)
            face = img[y1:y2, x1:x2]
            face = cv2.resize(face,(160,160))
    return face


def get_embedding(face):
    face = face.astype('float32')
    mean,std =  face.mean(), face.std()
    standardized = (face - mean) / std
    samples = np.expand_dims(standardized,axis=0)
    predict = face_model.predict(samples)[0]
    return predict



class MainPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.controller = controller
        self.svc_model, self.in_encoder, self.out_encoder = read_image_from_files()
        self.time_count = 0
        self.current_name = ''
        self.current_image = None  
        self.img_frame = None
        controller.protocol('WM_DELETE_WINDOW', self.destructor)
        self.panel = Label(self)  
        self.panel.pack(padx=10, pady=10)
        self.panel1 = Label(self)  
        self.panel1.pack(padx=10, pady=10)

        self.panel2 = Label(self) #Label(self, bg='#fff', fg='#eee')  
        self.panel2.pack()

        self.arr_data = []

        self.employee_id = StringVar()
        lbl_employee_id = Label(self.panel1, text = 'Select a person', font = ('calibre',10,'bold')) 
        self.textField_id = Entry(self.panel1,textvariable=self.employee_id,font=('calibre',10,'bold'))

        lbl_employee_id.grid(row=0,column=0)

        self.load_cbb()
        self.load_data_cbb()

        btnSaveImage = Button(self.panel2, text="Save Image", command=self.take_photo, width=15)
        btnSaveImage.grid(padx=15, pady=10,row=0,column=1)

        self.lbl_num_img = Label(self.panel2, text="User images", width=15, bg='#eee')
        self.lbl_num_img.grid(padx=0, pady=0,row=0,column=2)
        # btnSaveImage.pack(anchor=W, side=TOP, padx=15, pady=10)

        btnReloadData = Button(self.panel2, text="ReloadData", command=self.load_data_cbb, width=15)
        # btnReloadData.pack(anchor=E, side=TOP, padx=15 , pady=10)
        btnReloadData.grid(padx=15, pady=10,row=0,column=0)

        btnReloadTrainData = Button(self.panel2, text="Reload Images", command=self.read_load_from_DB, width=15)
        # btnReloadTrainData.pack(anchor=E, side=RIGHT, padx=15, pady=10)
        btnReloadTrainData.grid(padx=15, pady=10,row=1,column=0)

        btnQuit = Button(self.panel2, text="Quit", command=self.controller.quit, width=15)
        # btnQuit.pack(padx=15, pady=30)
        btnQuit.grid(padx=15, pady=10,row=1,column=1)


    def destructor(self):
        self.controller.destroy()
        self.vs.release()  
        cv2.destroyAllWindows()  

    def run_video(self):
        self.vs = cv2.VideoCapture(0)
        print(self.vs)


    def video_loop(self):
        # self.vs = cv2.VideoCapture(0)
        ok, frame = self.vs.read()
        self.img_frame = frame
        if ok:
            (h, w) = frame.shape[:2]
            text = ''
            
            try:
                
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
                        # get face
                        face = frame[y1:y2, x1:x2]
                        face = cv2.resize(face,(160,160))

                        # get embedding
                        current_embedding = get_embedding(face)
                        current_embedding = np.expand_dims(current_embedding,axis=0)
                        current_embedding_tranformed = self.in_encoder.transform(current_embedding)

                        # predict
                        predict = self.svc_model.predict(current_embedding_tranformed)   
                        predict_probability = self.svc_model.predict_proba(current_embedding_tranformed)    
                        idx = predict[0]
                        probability = predict_probability[0,idx] * 100
                        probability = round(probability, 4)

                        text = ''
                        
                        if probability  > 75:
                            name = str(self.out_encoder.inverse_transform(predict)[0])
                            age = 'unknown'
                            job = 'unknown'
                            for d in self.arr_data:
                                if d['name'] == name:
                                    age = d['age']
                                    job = d['job']

                            text = name + ' ' + str(probability) 
                            text2 = 'Age: ' + str(age) + ' Job: ' + str(job)

                            self.time_count +=1  
                            # print(text)
                            if self.time_count == 15 and self.current_name != name:
                                self.say(name)
                                self.current_name = name
                                self.time_count = 0
                        else:
                            text = 'unknown'
                            self.time_count = 0
                cv2.putText(frame, text, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, text2, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            except Exception as e:
                pass
            
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(image)  
            imgtk = ImageTk.PhotoImage(image=self.current_image)  
            self.panel.imgtk = imgtk 
            self.panel.config(image=imgtk)  
        self.controller.after(20, self.video_loop)

    def load_data_cbb(self):
        people = People.objects.all()

        values = [p.name for p in people]
        for p in people:
            data = {
                'name':p.name,
                'age':p.age,
                'job':p.job
            }
            self.arr_data.append(data)
        self.CBB['values'] = values

    def take_photo(self):
        user = self.CBB.get()
        if user:
            folder_user = folder + f'/{user}'
            if not os.path.exists(folder_user):
                os.makedirs(folder_user)
                numcount = 0
            else:
                path, dirs, files = next(os.walk(folder_user))
                numcount = len(files)
            image = self.img_frame
            text = f'{user}: {numcount} images'
            self.lbl_num_img.config(text = text)
            new_file_dir = folder_user + f'/{user}_{numcount}.jpg'
            cv2.imwrite(new_file_dir, image)

    def load_cbb(self):
        self.CBB = ttk.Combobox(self.panel1)
        try:
            self.CBB.current(0)
            self.CBB.bind('<<ComboboxSelected>>', self.cbb_onlected)
        except:
            pass
        self.CBB.grid(row=2)

    def read_load_from_DB(self):
        self.svc_model, self.in_encoder, self.out_encoder = read_image_from_files()

    def say(self,text):
        engine = pyttsx3.init()
        engine.setProperty('volume',1.0)
        engine.setProperty('rate',150)
        engine.say('xin chào ' +text)
        engine.runAndWait()

class App(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.geometry('700x700')
        container = Frame(self)
        container.pack()
        page = MainPage(parent=container,controller=self)
        page.grid(row=0,column=0,sticky='nsew')
        page.run_video()
        page.video_loop()
        page.tkraise()
        
class Command(BaseCommand):
    def handle(self, *args, **options):
        app = App()
        app.mainloop()
