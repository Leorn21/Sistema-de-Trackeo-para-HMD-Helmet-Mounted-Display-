import cv2
import numpy as np
import itertools 

# --- CONFIGURACIÓN ---
MIN_AREA = 50       
MAX_AREA = 1000 
MIN_POINTS = 5  
MAX_LINE_ERROR = 260 

# Frames maximos permitidos sin deteccion antes de reiniciar el tracking
MAX_LOST_FRAMES = 3  

# --- CLASE KALMAN ---
class LedKalman:
    def __init__(self, initial_point):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[initial_point[0]], [initial_point[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[initial_point[0]], [initial_point[1]], [0], [0]], np.float32)

    def predict(self):
        prediction = self.kalman.predict()
        return (int(prediction[0].item()), int(prediction[1].item()))

    def correct(self, point):
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        self.kalman.correct(measurement)
        return point
    
    @property
    def position(self):
        return (int(self.kalman.statePost[0].item()), int(self.kalman.statePost[1].item()))

def calcular_error_linea(p1, p2, p3):
    area = abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1])) * 0.5
    return area

# --- INICIO ---
lower_bound = np.array([117, 0, 80])
upper_bound = np.array([160, 165, 255])

# RUTA DEL VIDEO
cap = cv2.VideoCapture(r'E:\0Backup\Proyects\Proyecto Investigacion\Imagenes\Video_LEDs.mp4') 

if not cap.isOpened():
    print("Error al abrir video.")
    exit()

kalman_filters = []
initialized = False     
lost_frames_count = 0 # Contador para el timeout

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- Preprocesamiento ---
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Detección ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA and area < MAX_AREA and len(cnt) >= MIN_POINTS:
            ellipse = cv2.fitEllipse(cnt)
            candidates.append({'center': (int(ellipse[0][0]), int(ellipse[0][1])), 'ellipse': ellipse})

    # --- Tracking ---
    display_points = []
    status_msg = "BUSCANDO..."
    color_status = (0, 165, 255)

    if not initialized:
        # FASE DE BÚSQUEDA (Reiniciado)
        lost_frames_count = 0 
        if len(candidates) >= 3:
            min_error = float('inf')
            best_trio = []
            for combo in itertools.combinations(candidates, 3):
                p1, p2, p3 = combo[0]['center'], combo[1]['center'], combo[2]['center']
                err = calcular_error_linea(p1, p2, p3)
                if err < min_error:
                    min_error = err
                    best_trio = list(combo)
            
            if min_error < MAX_LINE_ERROR:
                best_trio.sort(key=lambda x: x['center'][0])
                kalman_filters = [LedKalman(led['center']) for led in best_trio]
                initialized = True
                print("Tracking Iniciado/Reiniciado.")

    else:
        # FASE DE SEGUIMIENTO
        predictions = [kf.predict() for kf in kalman_filters]

        matches = [None, None, None] 
        for i, pred in enumerate(predictions):
            closest = None
            min_dist = 50 
            for cand in candidates:
                dist = np.linalg.norm(np.array(pred) - np.array(cand['center']))
                if dist < min_dist:
                    min_dist = dist
                    closest = cand
            matches[i] = closest

        found_indices = [i for i, m in enumerate(matches) if m is not None]
        
        # --- Lógica de Timeout (Reinicio) ---
        if len(found_indices) < 2:
            # Si vemos menos de 2 LEDs, estamos perdiendo el track
            lost_frames_count += 1
            status_msg = f"PERDIENDO TRACK... ({lost_frames_count}/{MAX_LOST_FRAMES})"
            color_status = (0, 0, 255)
            
            if lost_frames_count > MAX_LOST_FRAMES:
                #Reiniciamos el sistema
                initialized = False
                kalman_filters = []
                print("Tracking perdido. Reiniciando búsqueda...")
        else:
            # Si vemos 2 o 3, el tracking es sólido
            lost_frames_count = 0
            
            if len(found_indices) == 3:
                status_msg = "TRACKING: SOLIDO (3/3)"
                color_status = (0, 255, 0)
                for i in range(3):
                    kalman_filters[i].correct(matches[i]['center'])
                    cv2.ellipse(frame, matches[i]['ellipse'], (0, 255, 0), 2)

            elif len(found_indices) == 2:
                status_msg = "TRACKING: INFERIDO (2/3)"
                color_status = (0, 255, 255)
                
                p_found = {i: matches[i]['center'] for i in found_indices}
                
                for idx in found_indices:
                    kalman_filters[idx].correct(matches[idx]['center'])
                    cv2.ellipse(frame, matches[idx]['ellipse'], (0, 255, 0), 2)

                missing_idx = [x for x in [0,1,2] if x not in found_indices][0]
                inferred_point = None

                if missing_idx == 2: 
                    vec = np.array(p_found[1]) - np.array(p_found[0])
                    inferred_point = tuple((np.array(p_found[1]) + vec).astype(int))
                elif missing_idx == 0: 
                    vec = np.array(p_found[1]) - np.array(p_found[2])
                    inferred_point = tuple((np.array(p_found[1]) + vec).astype(int))
                elif missing_idx == 1: 
                    mid = (np.array(p_found[0]) + np.array(p_found[2])) / 2
                    inferred_point = tuple(mid.astype(int))

                kalman_filters[missing_idx].correct(inferred_point)

        if initialized: # Solo dibujar si seguimos en modo tracking
            display_points = [kf.position for kf in kalman_filters]
            pts = np.array(display_points, np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
            for pt in display_points:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    cv2.putText(frame, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
    cv2.imshow('Tracking con Reinicio', frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()