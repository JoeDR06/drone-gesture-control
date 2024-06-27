import cv2

cam = cv2.VideoCapture(0)

while True:
    
    cap, frame = cam.read()
    
    if not cap:
        break
    
    cv2.imshow("frame",frame)
    
    if cv2.waitKey(1) in [27,32]:
        break
    
cam.release()
cv2.destroyAllWindows()