import cv2
import mediapipe as mp
import numpy as np
import datetime
import math
import tensorflow as tf
import requests
import json
import keyboard
import time
from tensorflow.keras.models import load_model
from collections import Counter
from PyQt5.QtCore import QUrl
from operator import rshift
from qt_third import ThirdPage
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from PyQt5.QtCore import QTimer, Qt, QTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import os

current_path = os.path.dirname(os.path.abspath(__file__))

model_lstm = load_model(os.path.join(current_path,'model/LSTM_model.h5'), compile=False)
model_mlp_1 = load_model(os.path.join(current_path,'model/MLP_model_1.h5'),compile=False)
model_mlp_2 = load_model(os.path.join(current_path,'model/MLP_model_2.h5'),compile=False)

lstm_labels_map = {0: 'normal', 1: 'abnormal'}
mlp_label_mapping = {0: 'D', 1: 'L', 2: 'U', 3: 'R', 4:'F'}
new_mlp_label_mapping ={0:'F', 1:'NF'}

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
RIGHT_IRIS = [474,475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33] # right eye right most landmark
L_H_RIGHT = [133] # right eye left most landmark
R_H_LEFT = [362] # left eye right most landmark
R_H_RIGHT = [263] # left eye left most landmark

lstm_label = []
mlp_label = []


form_class_secondpage = uic.loadUiType(os.path.join(current_path,'ui/Qt_designer_secondpage.ui'))[0]

class Secondpage(QMainWindow, form_class_secondpage):
    def __init__(self, student_id, name, subjects):
        super(Secondpage, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Cheat Block")
        self.setGeometry(650, 150, 1200, 1100)

        self.logo_image_path = "logo_sound/test_notice_logo.png"
        self.set_logo_image(self.logo_image_path)

        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.time_out_timer = QTimer()
        self.time_out_timer.timeout.connect(self.update_timer)

        self.seconds = 0
        self.start_button.clicked.connect(self.button1Function)
        self.finish_button.clicked.connect(self.button2Function)

        self.start_button.setEnabled(True)
        self.finish_button.setEnabled(False)

        self.show()

        self.window_data = None
        self.x_y_z_list = []
        self.event_list = []
        self.log_list = []
        self.cnt = 0
        self.previous = 0
        self.count = 0

        self.current_time = datetime.datetime.now()

        self.student_id = student_id
        self.name = name
        self.subjects = subjects

    def set_logo_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(1000, 750)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setScaledContents(True)

    def button1Function(self):
        self.cap = cv2.VideoCapture(0)
        self.update_frame()
        self.start_timer()
        self.start_button.setEnabled(False)
        self.finish_button.setEnabled(True)

    def button2Function(self):
        self.finish_test()

    def update_frame(self):
        ret, image = self.cap.read()
        if ret:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.face_mesh.process(image)
            image.flags.writeable = True
            image_eyecrop = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []

            x, y, z = 0, 0, 0

            iris_pos = ""

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            face_2d.append([x, y])

                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    rmat, jac = cv2.Rodrigues(rot_vec)

                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360


                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                     dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    cv2.line(image, p1, p2, (255, 0, 0), 3)
                    cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

                mesh_points = np.array(
                    [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv2.circle(image, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(image, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(image, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, mesh_points[L_H_RIGHT][0], 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, mesh_points[L_H_LEFT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)
                iris_pos, R_ratio, L_ratio = self.iris_position(center_right, center_left, mesh_points[R_H_RIGHT],
                                                                mesh_points[R_H_LEFT][0], mesh_points[L_H_RIGHT],
                                                                mesh_points[L_H_LEFT][0])
                if iris_pos == "":
                    iris_pos = "center"

                cv2.putText(image_eyecrop, "iris pos : {} R_ratio : {:.2f} L_ratio : {:.2f}".format(iris_pos, R_ratio, L_ratio),
                            (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv2.LINE_AA)
                
            self.x_y_z_list.append([x, y, z])

            if len(self.x_y_z_list) >= 150:
                lstm_predictions = []

                splitted_list = self.x_y_z_list[0:]
                self.window_data = np.array(splitted_list)
                self.window_data = self.window_data.reshape((1, self.window_data.shape[0], 3))
                lstm_predict_result = model_lstm.predict(self.window_data)

                if lstm_predict_result > 0.5:
                    lstm_predictions.append(1)
                    self.event_list.append(1)

                    self.log_list.append(['고개방향 이상탐지', self.seconds, self.current_time.strftime('%Y-%m-%d %H:%M:%S')])
                else:
                    lstm_predictions.append(0)
                    mlp_new_predictions = model_mlp_2.predict(self.window_data[0])
                    new_mlp_labels = [new_mlp_label_mapping[np.argmax(pred)] for pred in mlp_new_predictions]
                    most_newmlp_value = new_mlp_labels[-1]

                    self.event_list.append(0)
                    if most_newmlp_value == 'F' and (iris_pos == "left" or iris_pos == "right"):
                        self.Anomaly_detection_message_2()
                        self.log_list.append(["시선방향 이상탐지", self.seconds, self.current_time.strftime('%Y-%m-%d %H:%M:%S')])

                self.count_abnormal(self.event_list)

                counter_lstm = Counter(lstm_predictions)
                most_lstm_common_value = counter_lstm.most_common(1)[0][0]

                lstm_label.append(lstm_labels_map[most_lstm_common_value])

                self.x_y_z_list = self.x_y_z_list[30: 150]

            if len(lstm_label) > 0:
                cv2.putText(image, f'Class: {lstm_label[-1]}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            height, width, channel = image.shape
            bytesPerLine = 3 * width
            q_img = QtGui.QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            pixmap = QtGui.QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.width(), pixmap.height())

            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

            QtCore.QTimer.singleShot(1, self.update_frame)

    def start_timer(self):
        self.time_out_timer.start(1000)

    def update_timer(self):
        self.seconds += 1
        self.timer_label.setText(str(datetime.timedelta(seconds=self.seconds)))

        if self.seconds >= 7200:
            self.show_message()

    def show_message(self):
        msg = QMessageBox()
        msg.setWindowTitle("Timeout Alert")
        msg.setText("시험 시간이 모두 종료되었습니다!")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.time_out_timer.stop()
            self.cap.release()
            self.next_page = ThirdPage(log_list=self.log_list)
            self.next_page.show()
            self.close()

    def finish_test(self):
        msg = QMessageBox()
        msg.setWindowTitle("Finish Test")
        msg.setText("시험을 종료합니다.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.time_out_timer.stop()
            self.cap.release()
            self.next_page = ThirdPage(log_list=self.log_list)
            self.next_page.show()
            self.close()

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def iris_position(self, iris_center_right, iris_center_left, R_right_point, R_left_point, L_right_point,
                      L_left_point):
        R_center_to_right_dist = self.euclidean_distance(iris_center_right, R_right_point) - 1
        R_total_distance = self.euclidean_distance(R_right_point, R_left_point) - 3
        R_ratio = R_center_to_right_dist / R_total_distance
        L_center_to_right_dist = self.euclidean_distance(iris_center_left, L_right_point) - 1
        L_total_distance = self.euclidean_distance(L_right_point, L_left_point) - 3
        L_ratio = L_center_to_right_dist / L_total_distance
        iris_position = ""
        if R_ratio >= 0.65:
            iris_position = "left"
        elif L_ratio <= 0.35:
            iris_position = "right"
        elif 0.45 <= R_ratio <= 0.55:
            iris_position = "center"
        return iris_position, R_ratio, L_ratio

    def Anomaly_detection_message(self):
        msg = QMessageBox()
        msg.setWindowTitle("Anomaly Detection")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)

        mlp_predictions = model_mlp_1.predict(self.window_data[0])
        mlp_labels = [mlp_label_mapping[np.argmax(pred)] for pred in mlp_predictions]
        counter_mlp = Counter(mlp_labels)
        most_mlp_common_value = counter_mlp.most_common(1)[0][0]
        mlp_label.append(most_mlp_common_value)

        mlp_result = mlp_label[-1] if len(mlp_label) > 0 else "Unknown"
        msg.setText(f"이상행동이 감지되었습니다.\n 감지내용 : 고개방향 이상\n 감지 방향 : {mlp_result}")
        msg.exec_()

    def Anomaly_detection_message_2(self):
        msg = QMessageBox()
        msg.setWindowTitle("Anomaly Detection")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)

        msg.setText("이상행동이 감지되었습니다.\n감지내용 : 시선방향 이상")
        msg.exec_()

    def Anomaly_detection_sound(self):
        sound_path = "logo_sound/sound_1.wav"
        QSound.play(sound_path)

    def Anomaly_detection_sound_2(self):
        sound_path = "logo_sound/sound_2.wav"
        QSound.play(sound_path)

    def count_abnormal(self, lst):
        pre = self.previous

        tmp = lst[-1]
        if tmp == pre and pre == 1:
            self.previous = 1
        elif tmp == pre and pre == 0:
            self.previous = 0
        elif tmp != pre and pre == 0:
            self.count += 1
            cnt = self.count
            self.previous = 1
            if cnt == 1:
                self.Anomaly_detection_sound()
            elif cnt == 2:
                self.Anomaly_detection_sound_2()
            elif cnt >= 3:
                self.Anomaly_detection_message()
                self.kakaoapi("고개 방향 이상")
        elif tmp != pre and pre == 1:
            self.previous = 0

    def kakaoapi(self, str):
        with open("kakao/result_code.json", "r") as fp:
            tokens = json.load(fp)

        url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        headers = {
            "Authorization": "Bearer " + tokens['access_token']
        }
        data = {
            "template_object": json.dumps({"object_type": "text",
                                            "text": "{0} \n{1} {2}학생 부정행위 적발\n적발 내용 : {3}".format(self.subjects, self.student_id,
                                                                                      self.name, str),
                                            "link": {
                                                "web_url": "https://www.google.co.kr/search?q=drone&source=lnms&tbm=nws",
                                                "mobile_web_url": "https://www.google.co.kr/search?q=drone&source=lnms&tbm=nws"
                                            }
                                            })
        }
        response = requests.post(url, headers=headers, data=data)
        if response.json().get('result_code') == 0:
            print('메시지를 성공적으로 보냈습니다.')
        else:
            print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))

    def keyPressEvent(self, e):
        if keyboard.is_pressed('ctrl') & keyboard.is_pressed('c'):
            self.log_list.append(["Ctrl + C 입력 감지", self.seconds, self.current_time.strftime('%Y-%m-%d %H:%M:%S')])
            str = "Press Ctrl + C Key"
            print(str)
            time.sleep(0.5)

        elif keyboard.is_pressed('ctrl') & keyboard.is_pressed('v'):
            self.log_list.append(["Ctrl + V 입력 감지", self.seconds, self.current_time.strftime('%Y-%m-%d %H:%M:%S')])
            str = "Press Ctrl + V Key"
            print(str)
            time.sleep(0.5)

        elif keyboard.is_pressed('alt') & keyboard.is_pressed('tab'):
            self.log_list.append(["Alt + Tab 입력 감지", self.seconds, self.current_time.strftime('%Y-%m-%d %H:%M:%S')])
            str = "Press Alt + tab Key"
            print(str)
            time.sleep(0.5)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.log_list.append(["우클릭 입력 감지", self.seconds, self.current_time.strftime('%Y-%m-%d %H:%M:%S')])
            str = "Right mouse button pressed"
            print(str)
            time.sleep(0.5)

if __name__ == "__main__":
    app = QApplication([])
    window = Secondpage()
    window.show()
    app.exec_()
