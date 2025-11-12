import cv2
import numpy as np

# --- 1. Definir valores HSV ---
lower_bound = np.array([117, 0, 80])
upper_bound = np.array([160, 165, 255])

# --- 2. Inicia la captura de video ---
cap = cv2.VideoCapture(r'') #AGREGAR RUTA DEL VIDEO ENTRE LAS COMILLAS

if not cap.isOpened():
    print("Error: No se pudo abrir el archivo.")
    exit()

while True:
    # --- 3. Lee un fotograma
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir el fotograma. Finalizando stream")
        break

    # --- 4. Aplica lógica de detección por fotograma ---
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crea la máscara y limpia
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar y procesar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 170 and cv2.contourArea(cnt) < 900 and len(cnt) >= 5 :
            ellipse = cv2.fitEllipse(cnt)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    # --- 5. Resultados ---
    cv2.imshow('Video en Tiempo Real', frame)
    cv2.imshow('Mask', mask)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# --- 6. Liberar recursos ---
cap.release()
cv2.destroyAllWindows()

