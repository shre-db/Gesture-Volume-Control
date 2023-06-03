import cv2 as cv
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.max_hands,
            self.model_complexity,
            self.detection_conf,
            self.track_conf
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for idx, lm in enumerate(hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(idx, cx, cy)
                lm_list.append([idx, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)
        return lm_list


def main():
    p_time = 0
    c_time = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
