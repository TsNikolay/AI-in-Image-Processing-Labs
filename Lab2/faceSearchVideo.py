import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture("Lab2\data\Video.mp4")
while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (240, 425))
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_rects = face_cascade.detectMultiScale(gray_filter, 1.1, 5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
        roi_gray = gray_filter[y:y+h,x:x+w]
        roi_color= frame[y:y+h,x:x+w]
        smile= smile_cascade.detectMultiScale(roi_gray)
        eye= eye_cascade.detectMultiScale(roi_gray)

        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),1)
    
        for(ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)

        cv2.imshow('Video',frame)
    if (cv2.waitKey(1) & 0xFF==ord('q')):
        break


cap.release()
cv2.destroyAllWindows()


