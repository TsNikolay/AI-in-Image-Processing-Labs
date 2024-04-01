import cv2
import numpy as np

# Функція для малювання ліній на кадрі
def draw_lines(frame, lines, color=[0, 165, 255], thickness=7):
    if lines is None:
        return

    # Ініціалізація списків для координат лівих та правих ліній
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    # Перебір усіх ліній для визначення їх нахилу та класифікації
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:  # Перевірка на вертикальність лінії
                continue 
            slope = (y2 - y1) / (x2 - x1)  # Обчислення нахилу

            # Відкидання ліній, які мають надто малий нахил
            if abs(slope) < 0.5:
                continue

            # Класифікація ліній на ліві та праві
            if slope <= 0: 
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Визначення мінімальної та максимальної координати y для малювання ліній
    min_y = frame.shape[0] * (2.8 / 5)  
    max_y = frame.shape[0] 

    # Малювання лівої лінії, якщо вона була знайдена
    if len(left_line_x) > 0 and len(left_line_y) > 0:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        cv2.line(frame, (left_x_start, int(max_y)), (left_x_end, int(min_y)), color, thickness)

    # Малювання правої лінії, якщо вона була знайдена
    if len(right_line_x) > 0 and len(right_line_y) > 0:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        cv2.line(frame, (right_x_start, int(max_y)), (right_x_end, int(min_y)), color, thickness)

# Функція для обробки кадру
def process_image(frame):
    # Зміна розміру кадру
    frame = cv2.resize(frame, (400, 508))
    
    # Перетворення кадру в градації сірого
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Застосування Гауссівського розмиття
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    # Визначення країв за допомогою алгоритму Кенні
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    
    # Створення маски для виділення регіону інтересу
    vertices = np.array(
        [
            [
                (0, frame.shape[0]),
                (0, 350), 
                (100, 300),
                (frame.shape[1], 160), 
                (frame.shape[1], frame.shape[0])
            ]
        ], dtype=np.int32
    )
    
    # Застосування маски до кадру
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Використання перетворення Хафа для пошуку ліній
    rho = 3
    theta = np.pi / 180
    threshold = 100
    min_line_len = 75
    max_line_gap = 60
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    # Малювання ліній на кадрі
    draw_lines(frame, lines)
     
    return frame

# Отримання початкового зображення та його обробка
img = cv2.imread("Lab3/data/road.jpg")
img = cv2.resize(img, (400, 508))
processed_img = process_image(img)

# Додавання тексту до зображень
cv2.putText(img, "Original", (125,100), cv2.FONT_ITALIC, 1, (16,74,9), 4, cv2.LINE_AA)  
cv2.putText(processed_img, "Processed", (125,100), cv2.FONT_ITALIC, 1, (16,74,9), 4, cv2.LINE_AA)  

# Показ зображень
cv2.imshow('Image. Original', img)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow('Image. Processed', processed_img)
cv2.waitKey()
cv2.destroyAllWindows()

# Цикл для обробки відео
cap = cv2.VideoCapture("Lab3/data/Video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        processed_frame = process_image(frame)
        cv2.imshow('Video', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Закриття вікон і звільнення ресурсів
cap.release()
cv2.destroyAllWindows()
