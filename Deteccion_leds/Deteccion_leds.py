import cv2
import numpy as np
import itertools 

# --- CONFIGURACIÓN DE FILTROS ---
MIN_AREA = 50   
MAX_AREA = 1000 
MIN_POINTS = 5  
MAX_LINE_ERROR = 260 #Area maxima del triangulo formado por los 3 leds

# --- CLASE FILTRO DE KALMAN ---
class LedKalman:
    def __init__(self, initial_point):
        # Inicializamos un filtro de Kalman estandar de OpenCV
        # 4 variables de estado (x, y, dx, dy) -> Posicion y Velocidad
        # 2 variables de medicion (x, y) -> Solo medimos posicion
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # Matriz de Medicion (H): Relaciona estado con medicion
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                                  [0,1,0,0]], np.float32)
        
        # Matriz de Transicion (F): Como cambia el estado (x = x + dx)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]], np.float32)
        
        # Covarianza de Proceso (Q): Ruido del sistema (que tan suave es el movimiento)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Estado inicial
        self.kalman.statePre = np.array([[initial_point[0]], [initial_point[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[initial_point[0]], [initial_point[1]], [0], [0]], np.float32)

    def predict(self):
        # Predice la siguiente posicion basandose en la velocidad actual
        prediction = self.kalman.predict()
        return (int(prediction[0]), int(prediction[1]))

    def correct(self, point):
        # Corrige la prediccion con la medicion real (si existe)
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        self.kalman.correct(measurement)
        return point

# --- Calcular error de alineacion ---
def calcular_error_linea(p1, p2, p3):
    area = abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) * 0.5
    return area

# --- 1. Definir valores HSV ---
lower_bound = np.array([117, 0, 80])
upper_bound = np.array([160, 165, 255])

# --- 2. Inicia la captura de video ---
cap = cv2.VideoCapture(r'C:\Users\crist\Desktop\Proyecto Investigacion\Imagenes\Video_LEDs.mp4') 

if not cap.isOpened():
    print("Error: No se pudo abrir el archivo.")
    exit()


# Variables de estado del Sistema
kalman_filters = []     # Guarda 3 filtros (uno por LED)
initialized = False     # Bandera para saber si ya fijamos los LEDs

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    # --- 3. Preprocesamiento ---
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- 4. Detección de Candidatos ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA and area < MAX_AREA and len(cnt) >= MIN_POINTS:
            ellipse = cv2.fitEllipse(cnt)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            candidates.append({'center': center, 'ellipse': ellipse, 'area': area})

    # --- 5. FILTRO GEOMÉTRICO ---
    detected_leds = []
    
    # Logica para encontrar el trio valido
    if len(candidates) >= 3:
        if len(candidates) == 3:
            # Si hay 3, verificamos error
            error = calcular_error_linea(candidates[0]['center'], candidates[1]['center'], candidates[2]['center'])
            if error < MAX_LINE_ERROR:
                detected_leds = candidates
        else:
            # Si hay mas de 3, buscamos la mejor combinacion
            min_error = float('inf')
            best_combination = []
            for combo in itertools.combinations(candidates, 3):
                p1 = combo[0]['center']
                p2 = combo[1]['center']
                p3 = combo[2]['center']
                error = calcular_error_linea(p1, p2, p3)
                if error < min_error:
                    min_error = error
                    best_combination = combo
            
            if min_error < MAX_LINE_ERROR:
                detected_leds = list(best_combination)

    # --- 6. INTEGRACIÓN KALMAN ---
    display_points = [] # Puntos finales a dibujar
    is_prediction = False

    # ESCENARIO 1: Deteccion Exitosa (Tenemos 3 LEDs validos y alineados)
    if len(detected_leds) == 3:
        # Ordenamos por X para asegurar que el Kalman 0 sea el LED izquierdo
        detected_leds.sort(key=lambda x: x['center'][0])
        
        if not initialized:
            # Primera vez que vemos los LEDs: Inicializamos los 3 filtros
            kalman_filters = []
            for led in detected_leds:
                kf = LedKalman(led['center'])
                kalman_filters.append(kf)
            initialized = True
            print("Sistema Estabilizado: Tracking Iniciado.")

        # Actualizamos los filtros con las posiciones reales
        for i, led in enumerate(detected_leds):
            # 1. Predecir (siempre necesario)
            kalman_filters[i].predict()
            # 2. Corregir con el dato real
            corrected_point = kalman_filters[i].correct(led['center'])
            display_points.append(corrected_point) # Usamos el punto suavizado
        
        is_prediction = False

    # ESCENARIO 2: Fallo de Deteccion (Falso Negativo)
    else:
        if initialized:
            # Si ya habiamos iniciado, usamos Kalman para rellenar el hueco (PREDICCION)
            is_prediction = True
            for kf in kalman_filters:
                # Solo predecimos (sin correccion porque no hay dato)
                predicted_point = kf.predict()
                display_points.append(predicted_point)
        else:
            # Si aun no inicializamos y no hay 3 leds, no podemos hacer nada
            display_points = []

    # --- 7. DIBUJADO ---
    
    # 7.A Dibujar Elipses Reales (Si existen)
    for led in detected_leds:
        cv2.ellipse(frame, led['ellipse'], (0, 255, 0), 2)

    # 7.B Dibujar Resultado Kalman (Puntos y Líneas de Tracking)
    if len(display_points) == 3:
        pts = np.array(display_points, np.int32)
        
        # Color: Verde si es Deteccion Real, Rojo si es Prediccion (Memoria)
        color = (0, 0, 255) if is_prediction else (0, 255, 0)
        line_color = (0, 0, 255) if is_prediction else (255, 0, 0)

        # Dibujar linea
        cv2.polylines(frame, [pts], False, line_color, 2)

        # Dibujar puntos centrales (Tracking)
        for pt in display_points:
            cv2.circle(frame, pt, 5, color, -1)

        # Texto de estado
        status = "PREDICCION (MEMORIA)" if is_prediction else "TRACKING (REAL)"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Video con Kalman', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()