import numpy as np
import cv2

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
wc = 13
hc = 9
objp = np.zeros((wc * hc, 3), np.float32)
objp[:, :2] = np.mgrid[0:wc, 0:hc].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# 동영상 파일 경로 설정
video_file = './data/chessboard.mp4'

cap = cv2.VideoCapture(video_file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (wc, hc), None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)
        imgpoints.append(corners2)

        frame = cv2.drawChessboardCorners(frame, (wc, hc), corners2, ret)

        if len(objpoints) > 5:  # 충분한 프레임이 쌓이면 카메라 캘리브레이션 수행
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
            dst2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            cv2.imshow('Undistorted Frame', dst2)
        else:
            cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
