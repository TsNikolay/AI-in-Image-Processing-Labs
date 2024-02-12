import cv2
import imutils
import numpy 

# Зчитування та створення зображень
img = cv2.imread("Lab1\data\image.jpg")
imgGray = cv2.imread("Lab1\data\image.jpg",0)
imgCropped = img[145:620, 60:400]
imgResized = imutils.resize(img, width=300)
imgRotated = imutils.rotate(img, 45)
imgBlurred = cv2.GaussianBlur(img, (11,11), 0) # 11,11 - ступінь розмиття, 0 - стандартне відхилення (атоматично)
imgUnited = numpy.hstack((img, imgBlurred)) 
imgForRectangle = cv2.imread("Lab1\data\image.jpg")
imgForLine = cv2.imread("Lab1\data\image.jpg")
imgForPolylines = cv2.imread("Lab1\data\image.jpg")
imgForCircle = cv2.imread("Lab1\data\image.jpg")
imgForText = cv2.imread("Lab1\data\image.jpg")

# Промальовка необхідних фігур на зображенях
cv2.rectangle(imgForRectangle,(60,145),(400,620),(0,0,255),3) # зображення, координати, колір , товщина
cv2.line(imgForLine,(265,125),(255,355),(255,0,0),4) 
points = numpy.array([[185,160],[255,145],[335,175],[350,250],[325,320],[255,355],[180,310],[165,230]])
cv2.polylines(imgForPolylines, numpy.int32([points]),1,(0,255,0), 4) # 1 - вказівник на те, що треба закрити фігуру
cv2.circle(imgForCircle, (255,255), 90,(212,173,49), 3) # 255,265 - координати центра кола, 90 - радуіс 
font = cv2.FONT_ITALIC
cv2.putText(imgForText, "Person is found", (125,100) , font, 1 ,(16,74,9),4, cv2.LINE_AA) # 1 - масштабування, 

images = [img, imgGray, imgCropped,imgResized, imgRotated, imgBlurred, imgUnited,imgForRectangle,imgForLine,imgForPolylines,imgForCircle,imgForText]

#Послідовне відображення зображень
for image in images:
    cv2.imshow("Image", image)
    cv2.waitKey()

cv2.destroyAllWindows()



# Чорний фон
# img = numpy.zeros((200,200,3), np.uint8)

# Звернення до пікселю
# (b,g,r) = img[100,50]
# print("red =" , r , ", green =", g, ", blue =",b)

# Зміна розміру в cv2
# height,width,  = img.shape[0:2]
# ratio = width/height;
# newHeight = 300;
# newWidth = int(newHeight * ratio)
# imgResized = cv2.resize(img, (newWidth,newHeight))

# Поворот в cv2
# height, width = img.shape[:2]
# center = (width // 2, height//2)
# rotationMatrix = cv2.getRotationMatrix2D(center, 45, 1) # 45 - градуси повороту, 1 - зберегти розміри
# imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

# Зберігання файлу
# cv2.imwrite("Lab1\data\newImage.jpg",img)