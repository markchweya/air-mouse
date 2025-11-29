import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("opened:", cap.isOpened())

while True:
    ok, frame = cap.read()
    if not ok:
        print("read failed")
        break
    cv2.imshow("cam-test", frame)
    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
