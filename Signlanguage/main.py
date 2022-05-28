from kivy.app import App
from cProfile import label
from distutils.command.build import build
from os import remove
from turtle import update
from hashlib import new
from tkinter import CENTER
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import time
import mediapipe as mp
import numpy as np
from rsa import sign
from model.interpre import interpre
from kivy.properties import ObjectProperty, NumericProperty
import pywhatkit as kit

max_num_hands = 1

classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z'
        ]   
source1 = ObjectProperty()
fps = NumericProperty(30)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    
class Signlanguage(App):   
    def camscreen(self, *args):
        self.window.remove_widget(self.greeting)
        self.window.remove_widget(self.button)
        self.window.remove_widget(self.window.im)
        # self.window.im1 = Image(source="",
        #                         size_hint = (1,0.3))
        # self.window.add_widget(self.window.im1)
        # self.window.camera = Camera(play=True)
        # self.window.add_widget(self.window.camera)
        self.img1=Image()
        self.window.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
        # return layout
    
        # self.capture = cv2.VideoCapture(1)
        # self.my_camera = KivyCamera(capture=self.capture, fps=30)

        # return self.my_camera
    def update(self, dt):
        # display image from cam in opencv window
        ret, img = self.capture.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)

                # Gesture recognition model
                # interpreter = tf.lite.Interpreter(
                #     model_path='model/handtrain(400(91)).tflite',
                #     num_threads=1)
                # interpreter.allocate_tensors()
                # input_details = interpreter.get_input_details()
                # output_details = interpreter.get_output_details()
                # interpreter.set_tensor(input_details[0]['index'], data)
                # interpreter.invoke()
                # output_data = interpreter.get_tensor(output_details[0]['index'])
                # result=np.squeeze(output_data)
                result = interpre(data)
                idx = np.argmax(result)

                cv2.putText(img, text=classes[idx].upper(),
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        # cv2.imshow('Sign Language Recognition', img)
        # convert it to texture
        buf1 = cv2.flip(img, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1


    def build(self):

        self.window = GridLayout()
        self.window.im = Image(source="mini.jpg")
        self.window.cols =1
        self.window.size_hint = (0.3, 0.7)
        self.window.pos_hint = {"center_x":0.5, "center_y":0.5}
        self.window.add_widget(self.window.im)
        

        self.greeting = Label(text = "you want a  communicate? \n         touch the Button", font_size =12)
        self.window.add_widget(self.greeting)
        self.button = Button(text ="Start Translation", 
                             font_size=12, color = '#00FFCE', 
                             size_hint = (1,0.15), 
                             background_color = '#404040',
                             background_normal = "")
        self.button.bind(on_press=self.camscreen)
        self.window.add_widget(self.button)
        
        return self.window

    # def onButton(self, *args):
    #     return CameraScreen


        

if __name__ == "__main__":
    Signlanguage().run()