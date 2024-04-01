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

# 동영상 읽기
cap = cv2.VideoCapture('./data/chessboard.mp4')  # 동영상 파일 경로 입력

# 결과 저장을 위한 비디오 라이터 생성
out = cv2.VideoWriter('./data/output1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (int(cap.get(3)), int(cap.get(4))))

# 동영상 프레임별로 캘리브레이션 진행
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 코너를 찾았을 때만 포인트 저장
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너 시각화
        frame = cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

    # 결과 동영상에 프레임 저장
    out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()


# 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 출력
print("내부 파라미터 (fx, fy, cx, cy):")
print(mtx)
print("\n왜곡 계수:")
print(dist)

cv2.destroyAllWindows()

# 렌즈 왜곡 보정 수행
cap = cv2.VideoCapture('./data/chessboard.mp4')  # 동영상 파일 경로 입력
out = cv2.VideoWriter('./data/output2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 왜곡 보정
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, mtx)

    # 결과 동영상에 프레임 저장
    out.write(undistorted_frame)

    cv2.imshow('Undistorted Frame', undistorted_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
