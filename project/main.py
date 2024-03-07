import os
import sys
import pickle
import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qt_second import Secondpage #qt_second로 바꾸기
import sqlite3
import requests
import json
from mtcnn import MTCNN
from deepface import DeepFace
from deepface.commons import functions
from collections import Counter
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtGui import QFont
import icon_rc

current_path = os.path.dirname(os.path.abspath(__file__))

form_class_firstpage = uic.loadUiType(os.path.join(current_path, 'ui/Qt_designer_firstpage.ui'))[0]

class FirstPage(QMainWindow, form_class_firstpage):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Cheat Block")
        self.setGeometry(800, 60, 950, 1300)

        # 버튼 클릭 이벤트 연결
        self.btn_1.clicked.connect(self.button1Function)  # 시험 시작
        self.btn_2.clicked.connect(self.button2Function)  # 이미지 캡처

        self.btn_1.setEnabled(True)
        self.btn_2.setEnabled(False)

        #QLabel 참조변수
        self.label_time = self.findChild(QLabel, "label_time")

        #QLabel 이미지 설정
        self.set_logo_image("logo_sound/logo.png")
        
    def button1Function(self):
        self.btn_1.setEnabled(False)
        self.btn_2.setEnabled(True)        
        
        self.set_webcam_image()
        self.create_kakao_api()

    
    def create_kakao_api(self):
        # 카카오톡 메시지 API
        url = "https://kauth.kakao.com/oauth/token"
        data = {
            "grant_type": "refresh_token",
            "client_id": "2f8302848b284803ef29fc5157fef75d",
            "refresh_token": "HZOtcMB0nnTYiGQ8SnlwRr5VSUM_Gu-kYv2vvSvgCj1zmgAAAYh2fFwM"
        }
        response = requests.post(url, data=data)
        tokens = response.json()
        # kakao_code.json 파일 저장
        with open("kakao/result_code.json", "w") as fp:
            json.dump(tokens, fp)

    
    def set_logo_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(800, 800) # 크기 조절
        self.label_logo.setPixmap(scaled_pixmap)
        self.label_logo.setScaledContents(True)  # 이미지 크기에 맞게 조정
    
    def check_person_in_database(self,most_common_pred, database_path):
        # 데이터베이스 연결
        conn = sqlite3.connect(database_path)
        c = conn.cursor()

        c.execute("SELECT * FROM student WHERE label=?", (most_common_pred,))
        result = c.fetchone()

        if result is not None:
            # 학생 정보 추출
            id = result[0]
            self.student_id = result[2]
            self.name = result[1]

            # 수강 과목 정보 조회
            c.execute("SELECT subject FROM subject_student INNER JOIN subject ON subject_student.subject_id = subject.key WHERE id=?", (id,))
            self.subjects = c.fetchall()
            self.subjects = [subject[0] for subject in self.subjects]
            self.subjects_str = ", ".join(self.subjects)  # Convert list to a string with comma-separated values

            # 연결 종료
            conn.close()
            ui_student_id = self.lineEdit.text()
            ui_name = self.lineEdit_2.text()
            ui_subject_name = self.comboBox.currentText()
            if self.student_id == ui_student_id and self.name == ui_name and ui_subject_name in self.subjects:
                # UI에 입력된 값과 조회된 학생 정보가 일치하는 경우
                # 원하는 작업 수행
                print("Person is registered in the database.")
                
                # 얼굴 인식 및 추출
                out = DeepFace.extract_faces(self.frame,detector_backend = 'mtcnn')
                facial_area = out[0]["facial_area"]
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                
                frame = cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                # 텍스트 설정
                text_subject = "[" + ui_subject_name + "]" + ui_student_id + " " + ui_name
                text_certification = "본인 확인되었습니다."
                font_path = 'BMHANNAPro.ttf'
                font_size_subject = 30
                font_size_certification = 40
                text_color = (255, 255, 255)  # 흰색
                text_bg_color = (0, 0, 0)  # 검은색
                
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                font_subject = ImageFont.truetype(font_path, font_size_subject, encoding='utf-8')
                font_certification = ImageFont.truetype(font_path, font_size_certification, encoding='utf-8')

                # 텍스트 위치 계산
                text_size_subject = draw.textsize(text_subject, font=font_subject)
                text_width_subject, text_height_subject = text_size_subject
                x_subject = int((frame.shape[1] - text_width_subject) / 2)  # 가로 중앙
                y_subject = frame.shape[0] - text_height_subject - 45  # 하단 중앙
                
                # 텍스트 위치 계산
                text_size_certification = draw.textsize(text_certification, font=font_certification)
                text_width_certification, text_height_certification = text_size_certification
                x_certification = int((frame.shape[1] - text_width_certification) / 2)  # 가로 중앙
                y_certification = frame.shape[0] - text_height_certification - 5  # 하단 중앙

                # 텍스트 출력
                draw.rectangle((x_subject, y_subject, x_subject + text_width_subject, y_subject + text_height_subject), fill=text_bg_color)
                draw.text((x_subject, y_subject), text_subject, fill=text_color, font=font_subject)
                
                # 텍스트 출력
                draw.rectangle((x_certification, y_certification, x_certification + text_width_certification, y_certification + text_height_certification), fill=text_bg_color)
                draw.text((x_certification, y_certification), text_certification, fill=text_color, font=font_certification)
                
                # PIL 이미지를 OpenCV 이미지로 변환
                frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                

                height, width, channel = frame_with_text.shape
                bytesPerLine = 3 * width
                q_img = QImage(frame_with_text.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(300, 200)  # 크기 조절
                self.label_logo.setPixmap(scaled_pixmap)
                self.label_logo.setScaledContents(True)  # 이미지 크기에 맞게 조정
                QApplication.processEvents()
                self.show_message_3()

            else:
                self.cap.release()
                self.show_message()
        else: 
            self.cap.release()
            # 연결 종료
            conn.close()

            # 결과 반환 (None: 데이터베이스에 등록되지 않은 경우)
            self.show_message()   




    def set_webcam_image(self):
        self.cap = cv2.VideoCapture(0)

        while True:
            ret, self.frame = self.cap.read()
            if ret:
                self.frame = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)

                height, width, channel = self.frame.shape
                bytesPerLine = 3 * width
                q_img = QImage(self.frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(300, 200)  # 크기 조절
                self.label_logo.setPixmap(scaled_pixmap)
                self.label_logo.setScaledContents(True)  # 이미지 크기에 맞게 조정
                QApplication.processEvents()
            else:
                break
            
        self.cap.release()

    def button2Function(self):
        error_tmp = 0
        frame_np = np.array(self.frame)
        # df = DeepFace.find(img_path = frame_np, db_path = ".", detector_backend = 'mtcnn')
        try:
            df = DeepFace.find(img_path = frame_np, db_path = ".", detector_backend = 'mtcnn')
            df = df[0][df[0]["VGG-Face_cosine"] < 0.1]

            identity_list = df['identity'].tolist()
        except ValueError:
            identity_list = []
            error_tmp = 1
            self.show_message_4()

        if error_tmp == 1:
            most_common_identity = 'error'
        elif len(identity_list) == 0 :
            most_common_identity = "unknown"
        else:
            identity_list = [i.split('/')[-1].split('.')[0] for i in identity_list]

            # "_"를 기준으로 분할하여 첫 번째 요소를 가져옵니다.
            identity_prefixes = [i.split('_')[0] for i in identity_list]

            # 가장 많이 등장하는 요소를 찾습니다.
            counter = Counter(identity_prefixes)
            most_common_identity = counter.most_common(1)[0][0]
            
            if most_common_identity :
                print(most_common_identity)
            
        
        database_path = "C:/Backup/db/face_database.db"  # 데이터베이스 경로
        if most_common_identity != "error":
            self.check_person_in_database(most_common_identity, database_path)
        

    def show_message(self):
        msg = QMessageBox()
        msg.setWindowTitle("error")
        msg.setText("등록되지 않은 학생입니다.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.finished.connect(self.parent.reload_first_page)
        msg.exec_()
        self.close()
    
    def show_message_2(self):
        msg = QMessageBox()
        msg.setWindowTitle("error")
        msg.setText("얼굴 인식이 올바르게 되지 않았습니다. 다시 시도해주세요 ")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.finished.connect(self.parent.reload_first_page)
        msg.exec_()
        self.close()
        
    def show_message_3(self):
        msg = QMessageBox()
        msg.setWindowTitle("confirm")
        msg.setText("본인 인증이 확인되었습니다. ")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)  
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.cap.release()
            self.next_page = Secondpage(self.student_id, self.name, self.subjects)
            self.next_page.show()
            self.hide()
            
    def show_message_4(self):
        msg = QMessageBox()
        msg.setWindowTitle("error")
        msg.setText("얼굴이 정상적으로 검출되지 않았습니다.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.finished.connect(self.parent.reload_first_page)
        msg.exec_()
        self.close()                

        
class MainApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.first_page = FirstPage()
        self.first_page.parent = self

    def reload_first_page(self):
        self.first_page = FirstPage()
        self.first_page.parent = self
        self.first_page.show()


if __name__ == '__main__':
    app = MainApp(sys.argv)
    app.first_page.show()

    sys.exit(app.exec_())

