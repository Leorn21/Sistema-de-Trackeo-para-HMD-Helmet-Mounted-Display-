import argparse
import os
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# --- CONFIGURACIÓN ---
LED_COUNT = 6
MIN_FOUND = 3            # LEDs medidos minimos para considerar valido el tracking del frame
MAX_LOST_FRAMES = 12     # Frames seguidos sin tracking valido antes de reiniciar

# Deteccion
V_MIN = 160              # Brillo minimo (canal V). Los LEDs parpadean: apagados solo se ve el chip, mas tenue
S_MAX = 70               # Saturacion maxima (LED blanco)
TOPHAT_KERNEL = 45       # Top-hat: elimina zonas brillantes mas grandes que un LED (paredes, papeles)
TOPHAT_MIN = 60          # Contraste local minimo del LED respecto a su entorno
MIN_AREA = 12
MAX_AREA = 3000
MAX_ELONGATION = 4.0     # Relacion eje mayor/menor admitida (estela por desenfoque de movimiento)
MAX_CANDIDATES = 40

# Geometria de la barra (en pixeles o en "espaciados" entre LEDs vecinos)
MIN_SPACING = 12
MAX_SPACING = 250
SPACING_TOL = 0.3        # Tolerancia relativa al buscar el siguiente LED de la barra
MAX_INIT_ERROR = 0.15    # Error medio maximo (en espaciados) para aceptar una barra al inicializar
RESID_GATE = 0.35        # Distancia maxima medicion-prediccion tras compensar el movimiento del grupo
MIN_SHIFT_GATE = 0.5     # Salto del grupo siempre admitido (en espaciados), aunque Kalman no lo espere
MAX_NEIGHBOR_RATIO = 2.5 # Distancia entre filtros vecinos (en espaciados) que fuerza el reinicio
MIN_NEIGHBOR_RATIO = 0.3 # Dos filtros sobre el mismo LED tambien fuerzan el reinicio

# Kalman
KALMAN_MODEL = 'cv'      # 'cv' = velocidad constante, 'ca' = aceleracion constante
SIGMA_ACCEL = 3000.0     # px/s^2, ruido de aceleracion del modelo CV
SIGMA_JERK = 60000.0     # px/s^3, ruido de "jerk" del modelo CA
MEAS_NOISE = 2.0         # px, desviacion de la medicion del centro
INIT_VEL_STD = 800.0     # px/s, incertidumbre inicial de la velocidad
INIT_ACC_STD = 5000.0    # px/s^2, incertidumbre inicial de la aceleracion (solo CA)
INFERRED_NOISE_SCALE = 5.0  # Las posiciones inferidas por geometria pesan menos que las medidas
GATE_CHI2 = 13.8         # chi^2 con 2 g.l. al 99.9 %

DEFAULT_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Videos', 'Video_LEDs_nuevo.mp4')

# Rectangular: OpenCV lo separa en filas/columnas y es ~15x mas rapido que el eliptico a este tamaño
TOPHAT_SE = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL, TOPHAT_KERNEL))
OPEN_SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSE_SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# --- CLASE KALMAN ---
class LedKalman:
    """Kalman 2D por LED. El estado es [x, y, vx, vy] (CV) o [x, y, vx, vy, ax, ay] (CA)."""

    def __init__(self, initial_point, dt, model=KALMAN_MODEL, sigma_accel=SIGMA_ACCEL, sigma_jerk=SIGMA_JERK):
        # Bloques de un eje; con kron(., I2) se arma el estado intercalado [x, y, vx, vy, ...]
        if model == 'cv':
            F1 = np.array([[1, dt], [0, 1]])
            G1 = np.array([[dt**2 / 2], [dt]]) * sigma_accel        # aceleracion como ruido blanco
            P1 = np.diag([MEAS_NOISE**2, INIT_VEL_STD**2])
        elif model == 'ca':
            F1 = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
            G1 = np.array([[dt**3 / 6], [dt**2 / 2], [dt]]) * sigma_jerk  # jerk como ruido blanco
            P1 = np.diag([MEAS_NOISE**2, INIT_VEL_STD**2, INIT_ACC_STD**2])
        else:
            raise ValueError(f"Modelo de Kalman desconocido: {model}")

        n = 2 * F1.shape[0]
        I2 = np.eye(2)
        self.kalman = cv2.KalmanFilter(n, 2)
        self.kalman.transitionMatrix = np.kron(F1, I2).astype(np.float32)
        self.kalman.processNoiseCov = np.kron(G1 @ G1.T, I2).astype(np.float32)
        H = np.zeros((2, n), np.float32)
        H[0, 0] = H[1, 1] = 1
        self.kalman.measurementMatrix = H
        self.R = (MEAS_NOISE**2 * np.eye(2)).astype(np.float32)
        self.kalman.measurementNoiseCov = self.R.copy()
        # Incertidumbre inicial real: la velocidad de arranque no se conoce
        self.kalman.errorCovPost = np.kron(P1, I2).astype(np.float32)

        state = np.zeros((n, 1), np.float32)
        state[0], state[1] = initial_point
        self.kalman.statePost = state
        self.kalman.statePre = state.copy()

    def predict(self):
        prediction = self.kalman.predict()
        return np.array([prediction[0, 0], prediction[1, 0]])

    def innovation_cov(self):
        # S = H P- H^T + R: define la region donde se espera la proxima medicion
        return self.kalman.errorCovPre[:2, :2].astype(np.float64) + self.R

    def correct(self, point, noise_scale=1.0):
        self.kalman.measurementNoiseCov = self.R * np.float32(noise_scale**2)
        self.kalman.correct(np.array([[point[0]], [point[1]]], np.float32))
        self.kalman.measurementNoiseCov = self.R.copy()

    @property
    def position(self):
        return np.array([self.kalman.statePost[0, 0], self.kalman.statePost[1, 0]])


# --- DETECCION ---
def detectar_leds(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]

    # Blanco y brillante, y ademas mas brillante que su entorno inmediato (top-hat)
    color_mask = cv2.inRange(hsv, (0, 0, V_MIN), (179, S_MAX, 255))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, TOPHAT_SE)
    mask = cv2.bitwise_and(color_mask, cv2.inRange(tophat, TOPHAT_MIN, 255))
    # Primero cierre: une los fragmentos de los chips apagados y de las estelas partidas.
    # Despues una apertura chica, que limpia ruido sin borrar estelas finas
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_SE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_SE)

    # CHAIN_APPROX_NONE: los chips de los LEDs apagados son cuadrados y con SIMPLE quedan en 4 puntos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA or len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        ejes = ellipse[1]
        if max(ejes) > MAX_ELONGATION * max(min(ejes), 1e-3):
            continue

        # Centroide ponderado por intensidad: subpixel y estable en estelas
        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(roi_mask, [cnt - (x, y)], -1, 255, -1)
        pesos = v[y:y + h, x:x + w].astype(np.float32) * (roi_mask > 0)
        m = cv2.moments(pesos)
        if m['m00'] <= 0:
            continue
        center = (x + m['m10'] / m['m00'], y + m['m01'] / m['m00'])
        candidates.append({'center': center, 'ellipse': ellipse, 'score': m['m00']})

    # Ordenados por brillo total. No se recorta aca: en los frames con el LED apagado los chips son
    # chicos y quedarian fuera; el tracking filtra por region de busqueda y la busqueda inicial recorta
    candidates.sort(key=lambda c: c['score'], reverse=True)
    return candidates, mask


# --- GEOMETRIA DE LA BARRA ---
def buscar_barra(candidates):
    """Busca LED_COUNT candidatos alineados y equiespaciados recorriendo cadenas de vecinos.

    Devuelve los indices ordenados a lo largo de la barra, o None si no hay una barra clara.
    """
    candidates = candidates[:MAX_CANDIDATES]  # los mas brillantes, para acotar la busqueda
    if len(candidates) < LED_COUNT:
        return None
    pts = np.array([c['center'] for c in candidates])
    dists = np.linalg.norm(pts[:, None] - pts[None], axis=2)

    def extender(chain, errores):
        while len(chain) <= LED_COUNT:
            paso = pts[chain[-1]] - pts[chain[-2]]
            esperado = pts[chain[-1]] + paso
            d = np.linalg.norm(pts - esperado, axis=1)
            d[chain] = np.inf
            k = int(np.argmin(d))
            if d[k] > SPACING_TOL * np.linalg.norm(paso):
                break
            chain.append(k)
            errores.append(d[k] / np.linalg.norm(paso))

    best, best_err = None, MAX_INIT_ERROR
    vistos = set()
    for i in range(len(pts)):
        for j in np.argsort(dists[i])[1:5]:  # los LEDs vecinos suelen ser vecinos mas cercanos
            j = int(j)
            if (j, i) in vistos or not (MIN_SPACING <= dists[i, j] <= MAX_SPACING):
                continue
            vistos.add((i, j))
            chain, errores = [i, j], []
            extender(chain, errores)
            chain.reverse()
            extender(chain, errores)
            # Mas LEDs alineados de los esperados es ambiguo (patron repetitivo del fondo)
            if len(chain) != LED_COUNT:
                continue
            err = float(np.mean(errores))
            if err < best_err:
                best, best_err = chain, err

    if best is None:
        return None
    # Indice 0 = extremo izquierdo (o superior si la barra esta vertical)
    a, b = pts[best[0]], pts[best[-1]]
    if (b[0] - a[0], b[1] - a[1]) < (0, 0):
        best.reverse()
    return best


def parametros_barra(points):
    """Posicion de cada LED a lo largo de la barra, en espaciados (aprox. 0, 1, ..., N-1)."""
    points = np.asarray(points, np.float64)
    direccion = points[-1] - points[0]
    direccion /= np.linalg.norm(direccion)
    proy = (points - points[0]) @ direccion
    spacing = proy[-1] / (len(points) - 1)
    return proy / spacing, spacing


def ajustar_barra(t, points):
    """Ajuste por minimos cuadrados p_i = origen + t_i * paso. Devuelve (origen, paso)."""
    A = np.column_stack([np.ones_like(t), t])
    sol, *_ = np.linalg.lstsq(A, np.asarray(points, np.float64), rcond=None)
    return sol[0], sol[1]


# --- TRACKER ---
class BarTracker:
    def __init__(self, dt, model, sigma_accel, sigma_jerk):
        self.dt = dt
        self.model = model
        self.sigma_accel = sigma_accel
        self.sigma_jerk = sigma_jerk
        self.reset()

    def reset(self):
        self.filters = []
        self.t = None
        self.spacing = None
        self.lost_frames = 0

    @property
    def initialized(self):
        return bool(self.filters)

    def inicializar(self, candidates):
        idx = buscar_barra(candidates)
        if idx is None:
            return False
        points = [candidates[i]['center'] for i in idx]
        self.t, self.spacing = parametros_barra(points)
        self.filters = [LedKalman(p, self.dt, self.model, self.sigma_accel, self.sigma_jerk) for p in points]
        self.lost_frames = 0
        return True

    def estimar_desplazamiento(self, preds, cands, S):
        """Desplazamiento comun de la barra (cuerpo rigido) por votacion de hipotesis.

        Cada par (prediccion, candidato) propone un desplazamiento; gana el que alinea mas LEDs.
        Absorbe los saltos bruscos que el modelo de movimiento no anticipa.
        """
        gate_px = RESID_GATE * self.spacing
        hyps = np.vstack([np.zeros((1, 2)), (cands[None, :, :] - preds[:, None, :]).reshape(-1, 2)])
        S_inv = np.linalg.inv(S)
        m2 = np.einsum('hi,ij,hj->h', hyps, S_inv, hyps)
        validas = (m2 <= GATE_CHI2) | (np.linalg.norm(hyps, axis=1) <= MIN_SHIFT_GATE * self.spacing)
        hyps, m2 = hyps[validas], m2[validas]

        shifted = preds[None, :, :] + hyps[:, None, :]
        d = np.linalg.norm(shifted[:, :, None, :] - cands[None, None, :, :], axis=3).min(axis=2)
        inliers = (d < gate_px).sum(axis=1)
        costo = np.minimum(d, gate_px).sum(axis=1)
        # Mas LEDs alineados primero; a igualdad, el mas cercano a lo que esperaba Kalman
        orden = np.lexsort((m2, costo, -inliers))
        mejor = orden[0]
        if inliers[mejor] < 2:
            return np.zeros(2), 0
        return hyps[mejor], int(inliers[mejor])

    def actualizar(self, candidates):
        """Un paso de tracking. Devuelve (estado, indices medidos, puntos inferidos, desplazamiento)."""
        preds = np.array([kf.predict() for kf in self.filters])
        S = np.mean([kf.innovation_cov() for kf in self.filters], axis=0)

        # Region de busqueda: el salto maximo admitido mas el residuo por LED
        salto_max = max(np.sqrt(GATE_CHI2 * np.linalg.eigvalsh(S).max()), MIN_SHIFT_GATE * self.spacing)
        radio = salto_max + RESID_GATE * self.spacing
        all_cands = np.array([c['center'] for c in candidates]).reshape(-1, 2)
        cerca = np.linalg.norm(all_cands[:, None] - preds[None], axis=2).min(axis=1) <= radio
        cand_idx = np.flatnonzero(cerca)

        matched = {}
        shift = np.zeros(2)
        if len(cand_idx):
            cands = all_cands[cand_idx]
            shift, _ = self.estimar_desplazamiento(preds, cands, S)
            expected = preds + shift
            dist = np.linalg.norm(expected[:, None] - cands[None], axis=2)
            rows, cols = linear_sum_assignment(dist)
            for r, c in zip(rows, cols):
                if dist[r, c] < RESID_GATE * self.spacing:
                    matched[r] = int(cand_idx[c])

        # Rechazo de mediciones que no encajan en la recta de la barra
        while len(matched) > MIN_FOUND:
            ids = sorted(matched)
            pts = [candidates[matched[i]]['center'] for i in ids]
            origen, paso = ajustar_barra(self.t[ids], pts)
            resid = np.linalg.norm(np.array(pts) - (origen + np.outer(self.t[ids], paso)), axis=1)
            peor = int(np.argmax(resid))
            if resid[peor] < RESID_GATE * np.linalg.norm(paso):
                break
            del matched[ids[peor]]

        inferred = {}
        if len(matched) >= MIN_FOUND:
            self.lost_frames = 0
            ids = sorted(matched)
            pts = [candidates[matched[i]]['center'] for i in ids]
            origen, paso = ajustar_barra(self.t[ids], pts)
            self.spacing = float(np.clip(np.linalg.norm(paso), MIN_SPACING, MAX_SPACING))
            for i, kf in enumerate(self.filters):
                if i in matched:
                    kf.correct(candidates[matched[i]]['center'])
                else:
                    inferred[i] = origen + self.t[i] * paso
                    kf.correct(inferred[i], INFERRED_NOISE_SCALE)
            estado = 'solido' if len(matched) == LED_COUNT else 'inferido'
        else:
            self.lost_frames += 1
            for i, kf in enumerate(self.filters):
                if i in matched:
                    kf.correct(candidates[matched[i]]['center'])
                elif len(matched) >= 2:
                    # Pocas mediciones: al menos se arrastra el resto con el desplazamiento del grupo
                    kf.correct(preds[i] + shift, INFERRED_NOISE_SCALE)
            estado = 'perdiendo'

        if self.lost_frames > MAX_LOST_FRAMES:
            self.reset()
            return 'reinicio', matched, inferred, shift

        # La barra debe seguir siendo una barra: vecinos ni muy lejos ni encimados
        pos = np.array([kf.position for kf in self.filters])
        ratios = np.linalg.norm(np.diff(pos, axis=0), axis=1) / self.spacing
        if ratios.max() > MAX_NEIGHBOR_RATIO or ratios.min() < MIN_NEIGHBOR_RATIO:
            self.reset()
            return 'reinicio', matched, inferred, shift

        return estado, matched, inferred, shift


# --- DIBUJO ---
def dibujar(frame, candidates, tracker, matched, inferred, status_msg, color_status):
    for c in candidates:
        cv2.circle(frame, tuple(np.round(c['center']).astype(int)), 2, (160, 160, 160), -1)
    for c in matched.values():
        cv2.ellipse(frame, candidates[c]['ellipse'], (0, 255, 0), 2)
    for p in inferred.values():
        cv2.circle(frame, tuple(np.round(p).astype(int)), 8, (0, 255, 255), 1)
    if tracker.initialized:
        pts = np.round([kf.position for kf in tracker.filters]).astype(np.int32)
        cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
        for pt in pts:
            cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)
    cv2.putText(frame, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)


def procesar_video(video, model=KALMAN_MODEL, sigma_accel=SIGMA_ACCEL, sigma_jerk=SIGMA_JERK,
                   display=True, verbose=True, on_frame=None):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"Error al abrir video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = BarTracker(1.0 / fps, model, sigma_accel, sigma_jerk)

    m = {'frames': 0, 'solido': 0, 'inferido': 0, 'perdiendo': 0, 'buscando': 0,
         'inicializaciones': 0, 'reinicios': 0, 'saltos_compensados': 0, 'tiempo': 0.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        m['frames'] += 1
        t0 = time.perf_counter()

        candidates, _ = detectar_leds(frame)
        matched, inferred = {}, {}

        if not tracker.initialized:
            status_msg, color_status = "BUSCANDO...", (0, 165, 255)
            m['buscando'] += 1
            if tracker.inicializar(candidates):
                m['inicializaciones'] += 1
                if verbose:
                    print(f"[{m['frames']}] Tracking iniciado.")
        else:
            estado, matched, inferred, shift = tracker.actualizar(candidates)
            if tracker.spacing and np.linalg.norm(shift) > 0.25 * tracker.spacing:
                m['saltos_compensados'] += 1
            if estado == 'reinicio':
                m['reinicios'] += 1
                m['perdiendo'] += 1
                status_msg, color_status = "TRACK PERDIDO - REINICIANDO", (0, 0, 255)
                if verbose:
                    print(f"[{m['frames']}] Tracking perdido. Reiniciando busqueda...")
            else:
                m[estado] += 1
                if estado == 'solido':
                    status_msg, color_status = f"TRACKING: SOLIDO ({LED_COUNT}/{LED_COUNT})", (0, 255, 0)
                elif estado == 'inferido':
                    status_msg, color_status = f"TRACKING: INFERIDO ({len(matched)}/{LED_COUNT})", (0, 255, 255)
                else:
                    status_msg = f"PERDIENDO TRACK... ({tracker.lost_frames}/{MAX_LOST_FRAMES})"
                    color_status = (0, 0, 255)

        m['tiempo'] += time.perf_counter() - t0

        # Permite a otros scripts (p. ej. la evaluacion de error) leer el estado de cada frame
        if on_frame is not None:
            on_frame(m['frames'], frame, tracker, candidates, matched)

        if display:
            dibujar(frame, candidates, tracker, matched, inferred, status_msg, color_status)
            cv2.imshow('Tracking con Reinicio', frame)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()
    return m


def imprimir_metricas(m):
    total = m['frames']
    if total == 0:
        return
    pct = lambda k: 100.0 * m[k] / total
    print("\n--- Resultados del Tracking ---")
    print(f"Total de Frames: {total}")
    print(f"Tracking Solido: {m['solido']} frames ({pct('solido'):.2f}%)")
    print(f"Tracking Inferido: {m['inferido']} frames ({pct('inferido'):.2f}%)")
    print(f"Perdiendo track: {m['perdiendo']} frames ({pct('perdiendo'):.2f}%)")
    print(f"Buscando: {m['buscando']} frames ({pct('buscando'):.2f}%)")
    print(f"Inicializaciones: {m['inicializaciones']} | Reinicios por perdida: {m['reinicios']}")
    print(f"Saltos bruscos compensados: {m['saltos_compensados']}")
    print(f"Tiempo medio por frame: {1000 * m['tiempo'] / total:.1f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deteccion y tracking de la barra de LEDs")
    parser.add_argument('--video', default=DEFAULT_VIDEO)
    parser.add_argument('--model', choices=['cv', 'ca'], default=KALMAN_MODEL,
                        help="Modelo de Kalman: velocidad constante (cv) o aceleracion constante (ca)")
    parser.add_argument('--sigma-accel', type=float, default=SIGMA_ACCEL, help="Ruido de aceleracion (CV), px/s^2")
    parser.add_argument('--sigma-jerk', type=float, default=SIGMA_JERK, help="Ruido de jerk (CA), px/s^3")
    parser.add_argument('--no-display', action='store_true', help="Procesar sin ventana (para medir)")
    args = parser.parse_args()

    metricas = procesar_video(args.video, args.model, args.sigma_accel, args.sigma_jerk,
                              display=not args.no_display)
    imprimir_metricas(metricas)
