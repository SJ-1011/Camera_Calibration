import numpy as np
import cv2

# 체스보드의 가로 세로 내부 코너 수
chessboard_size = (13, 9)  # 예시로 9x6 체스보드 사용

# 체스보드 코너 좌표 생성
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 캘리브레이션을 위한 이미지 및 객체 포인트 저장을 위한 리스트 생성
objpoints = []  # 3D 객체 포인트
imgpoints = []  # 2D 이미지 포인트

# 이미지 경로 설정
images = ['./data/chessboard.jpg']  # 캘리브레이션 이미지들의 경로

# 이미지 읽어서 캘리브레이션 진행
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 코너를 찾았을 때만 포인트 저장
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너 시각화
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)  # 이미지 표시를 위한 대기시간 (500ms)
        
cv2.destroyAllWindows()

# 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("내부 파라미터 (fx, fy, cx, cy):")
print(mtx)
print("\n왜곡 계수:")
print(dist)

# 캘리브레이션을 위한 이미지를 output1로 내보내기
cv2.imwrite('output1.jpg', img)

# 왜곡 보정된 이미지 생성
img = cv2.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 결과 출력
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 왜곡 보정된 이미지를 output2로 내보내기
cv2.imwrite('output2.jpg', dst)
