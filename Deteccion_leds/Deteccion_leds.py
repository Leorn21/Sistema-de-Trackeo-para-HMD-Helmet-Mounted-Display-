import cv2
import numpy as np
import itertools 
from scipy.optimize import linear_sum_assignment  # <--- SÓLO SE AGREGA ESTO

# --- CONFIGURACIÓN ---
MIN_AREA = 20
MAX_AREA = 600
MIN_POINTS = 5  
MAX_LINE_ERROR = 200
LED_COUNT = 6

# Frames maximos permitidos sin deteccion antes de reiniciar el tracking
MAX_LOST_FRAMES = 6  

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

def calcular_error_linea(puntos):
    coords = np.array(puntos, dtype=np.float32)
    if len(coords) < 2:
        return float('inf')
    vx, vy, x0, y0 = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()
    denom = max((vx * vx + vy * vy) ** 0.5, 1e-6)
    diffs = coords - np.array([x0, y0], dtype=np.float32)
    dists = np.abs(diffs[:, 0] * vy - diffs[:, 1] * vx) / denom
    return float(dists.mean())

# --- INICIO ---
# LEDs blancos: muy brillante y baja saturacion
lower_bound = np.array([0, 0, 245])
upper_bound = np.array([179, 35, 255])

# RUTA DEL VIDEO
cap = cv2.VideoCapture(r'E:\0Backup\Proyects\Proyecto Investigacion\Deteccion_leds\Videos\Video_LEDs_nuevo.mp4') 

if not cap.isOpened():
    print("Error al abrir video.")
    exit()

kalman_filters = []
initialized = False     
lost_frames_count = 0 # Contador para el timeout
solid_tracking_frames = 0
inferred_tracking_frames = 0
total_frames = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    total_frames += 1

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
        if len(candidates) >= LED_COUNT:
            min_error = float('inf')
            best_set = []
            for combo in itertools.combinations(candidates, LED_COUNT):
                points = [item['center'] for item in combo]
                err = calcular_error_linea(points)
                if err < min_error:
                    min_error = err
                    best_set = list(combo)
            
            if min_error < MAX_LINE_ERROR:
                best_set.sort(key=lambda x: x['center'][0])
                kalman_filters = [LedKalman(led['center']) for led in best_set]
                initialized = True
                print("Tracking Iniciado/Reiniciado.")

    else:
        # FASE DE SEGUIMIENTO
        predictions = [kf.predict() for kf in kalman_filters]
        matches = [None] * LED_COUNT

        # --- SE INCOPORA EL ALGORITMO HÚNGARO MANTENIENDO TU LÓGICA ---
        if len(candidates) > 0:
            pred_arr = np.array(predictions, dtype=np.float32)
            cand_arr = np.array([c['center'] for c in candidates], dtype=np.float32)
            
            # Matriz de distancias
            dist_matrix = np.linalg.norm(pred_arr[:, None, :] - cand_arr[None, :, :], axis=2)
            
            # Resolución óptima global
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            
            # Se mantiene tu umbral original estricto de min_dist = 50 píxeles
            for p_idx, c_idx in zip(row_ind, col_ind):
                if dist_matrix[p_idx, c_idx] < 50:
                    matches[p_idx] = candidates[c_idx]
        # -------------------------------------------------------------

        found_indices = [i for i, m in enumerate(matches) if m is not None]
        
        # --- Lógica de Timeout (Reinicio) ---
        min_found = max(LED_COUNT - 2, 1)
        if len(found_indices) < min_found:
            # Si vemos menos de LEDs minimos, estamos perdiendo el track
            lost_frames_count += 1
            status_msg = f"PERDIENDO TRACK... ({lost_frames_count}/{MAX_LOST_FRAMES})"
            color_status = (0, 0, 255)
            
            if lost_frames_count > MAX_LOST_FRAMES:
                #Reiniciamos el sistema
                initialized = False
                kalman_filters = []
                print("Tracking perdido. Reiniciando búsqueda...")
        else:
            # Si vemos la mayoria, el tracking es sólido
            lost_frames_count = 0
            
            if len(found_indices) == LED_COUNT:
                status_msg = f"TRACKING: SOLIDO ({LED_COUNT}/{LED_COUNT})"
                color_status = (0, 255, 0)
                solid_tracking_frames += 1
                for i in range(LED_COUNT):
                    kalman_filters[i].correct(matches[i]['center'])
                    cv2.ellipse(frame, matches[i]['ellipse'], (0, 255, 0), 2)

            elif len(found_indices) >= min_found:
                status_msg = f"TRACKING: INFERIDO ({len(found_indices)}/{LED_COUNT})"
                color_status = (0, 255, 255)
                inferred_tracking_frames += 1

                for idx in found_indices:
                    kalman_filters[idx].correct(matches[idx]['center'])
                    cv2.ellipse(frame, matches[idx]['ellipse'], (0, 255, 0), 2)

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

# --- Métricas de Tracking ---
if total_frames > 0:
    percent_solido = (solid_tracking_frames / total_frames) * 100
    percent_inferido = (inferred_tracking_frames / total_frames) * 100
    print(f"\n--- Resultados del Tracking ---")
    print(f"Total de Frames: {total_frames}")
    print(f"Tracking Solido: {solid_tracking_frames} frames ({percent_solido:.2f}%) ")
    print(f"Tracking Inferido: {inferred_tracking_frames} frames ({percent_inferido:.2f}%)")