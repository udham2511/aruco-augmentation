import numpy
import cv2


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

shape = (1280, 720)

cap = cv2.VideoCapture(0)

cap.set(3, shape[0])
cap.set(4, shape[1])

video = cv2.VideoCapture(r"video\video.mp4")

video_shape = (video.get(3), video.get(4))

points_one = numpy.float32(
    [[0, 0], [0, video_shape[1]], [video_shape[0], video_shape[1]], [video_shape[0], 0]]
)

total_frames = video.get(7)
frame_count = 0

while cap.isOpened() and video.isOpened():
    success, frame = cap.read()

    success, video_frame = video.read()

    corners, ids, _ = cv2.aruco.detectMarkers(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dictionary
    )

    if ids is not None:
        mask = numpy.zeros_like(frame)

        for corner in corners:
            points_two = numpy.float32(corner)

            matrix = cv2.getPerspectiveTransform(points_one, points_two)
            mask += cv2.warpPerspective(video_frame, matrix, shape)

            cv2.fillConvexPoly(frame, numpy.int32(corner), 0)

        frame = cv2.add(frame, mask)

    frame_count += 1

    if frame_count == total_frames:
        frame_count = 0
        video.set(1, frame_count)

    cv2.imshow("Aruco Augmentation", cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cap.release()
cv2.destroyAllWindows()
