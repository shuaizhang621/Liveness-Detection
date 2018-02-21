# coding:utf-8
import os
import dlib
import tkinter as tki
import threading
import time
import imutils
import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk
from imutils import face_utils
from imutils.video import VideoStream

class LivenesDetectionApp:
    def __init__(self, detector, predictor):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.detector = detector
        self.predictor = predictor
        self.nose_p = []
        self.name = []
        self.frame = None
        self.vs = VideoStream().start()

        self.notEnd = True
        self.thread = None
        self.stopEvent = None

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.who = None
        self.preWho = None
        self.is_human = False

        self.stdstr = "std = ..."
        self.nose_std = 0

        # steps
        self.flag = 0
        self.StepOnePass = False
        self.StepTwoPass = False
        self.StepThreePass = False

        # face status
        self.leftEye_status = "open"
        self.rightEye_status = "open"
        self.month_status = "close"

        # counters
        self.RecCounter = 0
        self.BlinkCounter = 0

        # the factor whickh judge the fave change
        self.criterion_eye = 0.25
        self.criterion_mouth = 0.03

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create a button, that when pressed, will take the current
        # frame and save it to file

        btn1 = tki.Button(self.root, text="*close*", command=self.onClose)
        btn1.pack(side="bottom", fill="both", expand="no", padx=20, pady=5)

        btn2 = tki.Button(self.root, text="*Restart*", command=self.restart)
        btn2.pack(side="bottom", fill="both", expand="no", padx=20, pady=5)

        btn3 = tki.Button(self.root, text="*Next*", command=self.next)
        btn3.pack(side="bottom", fill="both", expand="no", padx=20, pady=5)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.setDaemon(True)
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("Human Verification")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):

        # read the training set
        avgImg, faceVec, diffTrain = self.ReconginitionVector(selecthr=0.9)
        try:
            while not self.stopEvent.is_set():
                time.sleep(0.1)
                # read frames
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=900)
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale frame
                rects = self.detector(gray, 0)
                if len(rects) == 0:
                    cv2.putText(self.frame, 'Please stay in the camera.', (300, 50), self.font, 0.8, (0, 0, 255), 2,
                                lineType=8)
                    self.nose_p = []
                    self.nose_std = 0
                    self.is_human = False
                else:
                    # loop over the face detections
                    for rect in rects:
                        # rectangle the face

                        judgeImg = gray[rect.top():rect.bottom(), rect.left():rect.right()]
                        try:
                            judgeImg = cv2.resize(judgeImg, (100, 100))
                        except:
                            judgeImg = cv2.resize(PrevJudgeImg, (100, 100))
                        PrevJudgeImg = judgeImg

                        # determine the facial landmarks for the face region
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # the coordinates of the face
                        right_eye = shape[36:42]
                        left_eye = shape[42:48]
                        mouth_up = tuple(shape[63])
                        mouth_down = tuple(shape[67])
                        left_eyebrow = tuple(shape[20])
                        right_eyebrow = tuple(shape[25])
                        jaw = tuple(shape[9])
                        left_cheek = tuple(shape[2])
                        right_cheek = tuple(shape[14])
                        nose = tuple(shape[30])
                        nose_1 = tuple(shape[29])
                        nose_2 = tuple(shape[33])

                        # calculate the distance of eyes, cheek, mouth
                        dst_mouth = mouth_up[1] - mouth_down[1]
                        dst_cheek = abs(right_cheek[0] - left_cheek[0])
                        nose_position = (nose[0] - left_cheek[0]) / (dst_cheek * 1.0)
                        self.nose_p.append(nose_position)
                        facelength = float((left_eyebrow[1] + right_eyebrow[1]) / 2 - jaw[1])


                        left_eye_ratio = self.eye_aspect_ratio(left_eye)
                        right_eye_ratio = self.eye_aspect_ratio(right_eye)
                        eye_radio = (left_eye_ratio + right_eye_ratio) / 2
                        mouth_ratio = dst_mouth / facelength

                        # step1 recognize the face
                        if self.flag == 0:
                            cv2.putText(self.frame, 'Step 1: face Recoginition', (5, 20), self.font, 0.5, (0, 0, 0), 2,
                                        lineType=8)
                            cv2.rectangle(self.frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()),
                                          (0, 255, 255), 2)
                            self.stepOne(judgeImg, faceVec, avgImg, diffTrain, rect)
                            if self.StepOnePass:
                                 cv2.putText(self.frame, 'Hello ' + self.who + '! Now Press Next.', (5, 50), self.font,
                                            1, (0, 255, 0), 2, lineType=8)
                            else:
                                cv2.putText(self.frame, 'Please look at the camera.', (5, 50), self.font, 1,
                                            (0, 0, 255), 2, lineType=8)

                        # step2 recognize the blink of eyes
                        if self.flag == 1:
                            # Step 2
                            for (x, y) in shape:
                                cv2.circle(self.frame, (x, y), 2, (255, 0, 0), 2)
                            self.stepTwo(mouth_ratio, eye_radio, tuple(shape[54]), tuple(shape[46]))
                            cv2.putText(self.frame, 'Step 2: eyes and mouth detection', (5, 20), self.font, 0.5,
                                        (0, 0, 0), 2, lineType=8)
                            if self.StepTwoPass:
                                cv2.putText(self.frame, 'Good Job! Now Press Next.', (5, 55),
                                            self.font, 1, (0, 255, 0), 2, lineType=8)
                            else:
                                cv2.putText(self.frame, 'Please Blink eyes and open(close) mouth.', (5, 55),
                                            self.font, 1, (0, 0, 255), 2, lineType=8)
                                blkCounter = str(np.floor(self.BlinkCounter))
                                cv2.putText(self.frame, blkCounter, (15, 100), self.font, 1, (0, 0, 0), 2, lineType=8)

                        # Step 3 Verify by nose position
                        if self.flag == 2:
                            self.stepThree(left_cheek, right_cheek, nose_1, nose_2, nose)
                            cv2.putText(self.frame, self.stdstr, (5, 110), self.font, 0.5, (255, 255, 0), 1, lineType=8)

                            if self.StepThreePass:
                                cv2.putText(self.frame, 'Step 3: Detect the nose move', (5, 20), self.font, 0.5, (0, 0, 0), 2, lineType=8)
                                cv2.putText(self.frame, 'Congratulations! You have passed the test!.',
                                            (5, 50), self.font, 1, (0, 255, 0), 2, lineType=8)
                            else:
                                cv2.putText(self.frame, 'Step 3:', (5, 20), self.font, 0.5, (0, 0, 255), 2, lineType=8)
                                cv2.putText(self.frame, 'Please turn your face to different directions', (5, 50),
                                            self.font, 1, (0, 255, 0), 2, lineType=8)
                                if len(self.nose_p) > 30:
                                    cv2.putText(self.frame, 'Please Try Again', (5, 80), self.font, 1, (0, 0, 255), 2,
                                                lineType=8)
                                else:
                                    cv2.putText(self.frame, 'Collecting Data...', (5, 80), self.font, 1,
                                                (0, 0, 0), 2,
                                                lineType=8)

                self.buildPanel()
        except:
            print("error")
            self.onClose()

    def stepOne(self, judgeImg, faceVec, avgImg, diffTrain, rect):
        judgeNum, resVal = self.judgeFace(np.mat(judgeImg).flatten(), faceVec, avgImg, diffTrain)
        cv2.putText(self.frame, self.name[judgeNum - 1][:-4], (rect.right(), rect.top() - 15), self.font, 1,
                    (255, 0, 0), 1,
                    lineType=8)

        if judgeNum in range(1, 21):
            self.who = "zhen"
            self.RecCounter += 1
        elif judgeNum in range(21, 41):
            self.who = "zhen"
            self.RecCounter += 1
        else:
            self.RecCounter = int(self.RecCounter / 2)

        if self.who != self.preWho:
            self.RecCounter = int(self.RecCounter / 2)
        self.preWho = self.who

        if self.RecCounter >= 30:
            self.StepOnePass = True
        elif self.RecCounter in range(0, 10):
            cv2.putText(self.frame, '.', (rect.left(), rect.top() - 10), self.font, 1, (0, 0, 255), 2,
                        lineType=8)
        elif self.RecCounter in range(10, 20):
            cv2.putText(self.frame, '..', (rect.left(), rect.top() - 10), self.font, 1, (0, 255, 255), 2,
                        lineType=8)
        elif self.RecCounter in range(20, 30):
            cv2.putText(self.frame, '...', (rect.left(), rect.top() - 10), self.font, 1, (0, 255, 0), 2,
                        lineType=8)

    def stepTwo(self, mouth_ratio, eye_ratio, mouth_side, eye_side):
        # Show eyes and mouth status in real time
        if mouth_ratio >= self.criterion_mouth:
            cv2.putText(self.frame, 'Mouth Open', mouth_side, self.font, 0.5, (0, 255, 0), 2, lineType=8)
            if self.month_status == "close":
                self.month_status = "open"
                self.BlinkCounter += 0.5
        else:
            cv2.putText(self.frame, 'Mouth Close', mouth_side, self.font, 0.5, (0, 0, 255), 2, lineType=8)
            if self.month_status == "open":
                self.month_status = "close"
                self.BlinkCounter += 0.5
        if eye_ratio > self.criterion_eye:
            cv2.putText(self.frame, 'Eye open', eye_side, self.font, 0.5, (0, 255, 0), 2,
                        lineType=8)
            if self.leftEye_status == 'close':
                self.leftEye_status = 'open'
                self.BlinkCounter += 0.34
        else:
            cv2.putText(self.frame, 'Eye close', eye_side, self.font, 0.5, (0, 0, 255), 2,
                        lineType=8)
            if self.leftEye_status == 'open':
                self.leftEye_status = 'close'
                self.BlinkCounter += 0.34
        if self.BlinkCounter >= 10:
            self.StepTwoPass = True

    def stepThree(self, left_cheek, right_cheek, nose_1, nose_2, nose):
        cv2.line(self.frame, right_cheek, left_cheek, (0, 255, 0), 2)
        cv2.line(self.frame, nose_1, nose_2, (255, 0, 0), 2)
        cv2.circle(self.frame, right_cheek, 2, (0, 255, 0), 2)
        cv2.circle(self.frame, left_cheek, 2, (0, 255, 0), 2)
        cv2.circle(self.frame, nose, 3, (0, 0, 255), 5)
        cv2.circle(self.frame,
                   (int((right_cheek[0] + left_cheek[0]) / 2), int((right_cheek[1] + left_cheek[1]) / 2)),
                   3, (0, 0, 255), 5)

        if len(self.nose_p) % 30 == 0:
            # calculate the matrix's Standard Deviation
            nose_std = np.std(self.nose_p)
            self.stdstr = "std = %f" % (nose_std)
            nose_p = []
            if nose_std > 0.12:
                self.StepThreePass = True

        cv2.putText(self.frame, self.stdstr, (5, 110), self.font, 0.5, (255, 255, 0), 1, lineType=8)

    def restart(self):
        self.flag = 0
        self.StepOnePass = False
        self.StepTwoPass = False
        self.StepThreePass = False
        self.RecCounter = 0
        self.BlinkCounter = 0

    def next(self):
        if self.StepOnePass and not self.StepTwoPass and not self.StepThreePass:
            self.flag = 1
        elif self.StepTwoPass and not self.StepThreePass:
            self.flag = 2
            self.nose_std = 0
            self.nose_p = []

    def buildPanel(self):
        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format

        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        # if the panel is None, we need to initialize it
        if self.panel is None:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)

        # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        radio = (A + B) / (2.0 * C)
        return radio

    def loadImageSet(self, add):
        FaceMat = np.mat(np.zeros((40, 10000)))
        j = 0
        for i in sorted(os.listdir(add)):
            pic = i.split('.')[1]
            if pic == 'jpg':
                try:
                    self.name.append(i)
                    print(i)
                    img = cv2.imread(add + i, 0)
                except:
                    print("load %s failed" % i)
                # flatten 折叠成一围数组
                FaceMat[j, :] = np.mat(img).flatten()
                j += 1
        return FaceMat

    def ReconginitionVector(self, selecthr=0.8):
        # step1: load the face image data ,get the matrix consists of all image
        FaceMat = self.loadImageSet('dataset/').T
        # step2: average the FaceMat
        avgImg = np.mean(FaceMat, 1)
        # step3: calculate the difference of avgImg and all image data(FaceMat)
        diffTrain = FaceMat - avgImg
        # step4: calculate EigenValue and EigenVector of covariance matrix
        eigvals, eigVects = np.linalg.eig(np.mat(diffTrain.T * diffTrain))
        # pick part of EigenVector according the EigenValue
        eigSortIndex = np.argsort(-eigvals)
        for i in range(np.shape(FaceMat)[1]):
            if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
                eigSortIndex = eigSortIndex[:i]
                break
        # calculate the lower dimension matrix
        # covVects is the eigenvector of covariance matrix
        faceVec = diffTrain * eigVects[:, eigSortIndex]
        return avgImg, faceVec, diffTrain

    def judgeFace(self, judgeImg, faceVec, avgImg, diffTrain):
        diff = judgeImg.T - avgImg
        weiVec = faceVec.T * diff
        res = 0
        resVal = np.inf
        for i in range(40):
            TrainVec = faceVec.T * diffTrain[:, i]
            # 求欧式距离
            dist = np.linalg.norm(weiVec - TrainVec)
            if dist < resVal:
                res = i
                resVal = dist
        if resVal > 5e+15:
            return -1, resVal
        else:
            return res + 1, resVal

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()


if __name__ == "__main__":
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # start the app
    pba = LivenesDetectionApp(detector, predictor)
    pba.root.mainloop()
