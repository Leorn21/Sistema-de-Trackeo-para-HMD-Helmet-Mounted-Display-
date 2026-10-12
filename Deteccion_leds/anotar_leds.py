"""Herramienta para anotar a mano el centro real de cada LED (verdad de terreno).

Uso:
    python anotar_leds.py --video ../Videos/Video_LEDs_nuevo.mp4 --n 40

Por cada frame elegido:
  1. Recuadrar con el mouse la zona de la barra y confirmar con ENTER (C = saltar el frame).
  2. En la vista ampliada, hacer clic en el centro de cada LED, de IZQUIERDA a DERECHA.
     X = LED no visible (se registra como faltante), U = deshacer, R = rehacer el frame,
     ENTER/ESPACIO = aceptar (con los 6 LEDs marcados), S = saltar el frame, Q = guardar y salir.

El CSV se guarda despues de cada frame; si se vuelve a ejecutar, continua donde quedo.
No se muestra el resultado del algoritmo para no condicionar la anotacion.
"""
import argparse
import csv
import os

import cv2
import numpy as np

from Deteccion_leds import DEFAULT_VIDEO, LED_COUNT

MAX_VIEW_W = 1280      # ancho maximo de la vista completa
ZOOM_VIEW_W = 1200     # ancho aproximado de la vista ampliada
MAX_ZOOM = 12


def frames_a_anotar(total, n, margen=10):
    """n frames repartidos uniformemente (numeracion desde 1, igual que procesar_video)."""
    return sorted(set(np.linspace(margen, total - 1, n).round().astype(int).tolist()))


def leer_anotados(path, solo_con_leds=False):
    """Frames ya registrados. Con solo_con_leds, los salteados (sin ningun LED marcado) quedan pendientes."""
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        filas = list(csv.DictReader(f))
    if solo_con_leds:
        return {int(r['frame']) for r in filas if r['x']}
    return {int(r['frame']) for r in filas}


def compactar(path):
    """Deja una sola anotacion por frame (la ultima), por si un frame salteado se volvio a anotar."""
    with open(path, newline='') as f:
        filas = list(csv.DictReader(f))
    por_frame = {}
    for r in filas:
        por_frame.setdefault(r['frame'], {})[r['led']] = r  # las filas posteriores reemplazan a las anteriores
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'led', 'x', 'y'])
        w.writeheader()
        for frame in sorted(por_frame, key=int):
            for led in sorted(por_frame[frame], key=int):
                w.writerow(por_frame[frame][led])


def anotar_frame(frame, n_frame):
    """Devuelve una lista de LED_COUNT puntos (x, y) o None por LED; 'skip' o 'quit'."""
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_VIEW_W / w)
    vista = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.putText(vista, f"Frame {n_frame}: recuadrar la barra y ENTER (C = saltar)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    x, y, rw, rh = cv2.selectROI('Anotar', vista, showCrosshair=False)
    if rw == 0 or rh == 0:
        return 'skip'
    x0, y0 = int(x / scale), int(y / scale)
    x1, y1 = min(w, int((x + rw) / scale) + 1), min(h, int((y + rh) / scale) + 1)
    roi = frame[y0:y1, x0:x1]
    zoom = max(1, min(MAX_ZOOM, ZOOM_VIEW_W // max(1, roi.shape[1])))
    # INTER_NEAREST: cada pixel original queda como un bloque, se ve exactamente donde cae el clic
    base = cv2.resize(roi, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

    puntos = []

    def a_original(px, py):
        # Centro del pixel ampliado -> coordenadas de pixel original (centro del pixel i = i)
        return x0 + (px + 0.5) / zoom - 0.5, y0 + (py + 0.5) / zoom - 0.5

    def on_mouse(event, px, py, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < LED_COUNT:
            puntos.append(a_original(px, py))

    cv2.namedWindow('Zoom')
    cv2.setMouseCallback('Zoom', on_mouse)
    while True:
        img = base.copy()
        for i, p in enumerate(puntos):
            if p is None:
                continue
            cx, cy = int(round((p[0] - x0 + 0.5) * zoom - 0.5)), int(round((p[1] - y0 + 0.5) * zoom - 0.5))
            cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 2 * zoom + 6, 1)
            cv2.putText(img, str(i), (cx + 4, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        faltan = LED_COUNT - len(puntos)
        msg = (f"LED {len(puntos)} (izq->der): clic en el centro | X faltante | U deshacer | S saltar"
               if faltan else "ENTER aceptar | U deshacer | R rehacer")
        cv2.putText(img, msg, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('Zoom', img)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('u'), ord('U')) and puntos:
            puntos.pop()
        elif k in (ord('x'), ord('X')) and faltan:
            puntos.append(None)
        elif k in (ord('r'), ord('R')):
            puntos.clear()
        elif k in (ord('s'), ord('S')):
            cv2.destroyWindow('Zoom')
            return 'skip'
        elif k in (ord('q'), ord('Q')):
            cv2.destroyWindow('Zoom')
            return 'quit'
        elif k in (13, 32) and not faltan:
            cv2.destroyWindow('Zoom')
            return puntos


def main():
    parser = argparse.ArgumentParser(description="Anotacion manual del centro de los LEDs")
    parser.add_argument('--video', default=DEFAULT_VIDEO)
    parser.add_argument('--n', type=int, default=40, help="Cantidad de frames a anotar")
    parser.add_argument('--out', help="CSV de salida (por defecto anotaciones/<video>.csv)")
    parser.add_argument('--salteados', action='store_true',
                        help="Volver a pedir los frames salteados (guardados sin ningun LED marcado)")
    parser.add_argument('--demo', action='store_true',
                        help="Modo demostracion: recorre todos los frames y no guarda nada (el CSV no se toca)")
    args = parser.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anotaciones',
                                   os.path.splitext(os.path.basename(args.video))[0] + '.csv')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Error al abrir video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.demo:
        pendientes = frames_a_anotar(total, args.n)
        print(f"MODO DEMOSTRACION: {len(pendientes)} frames. Las marcas se muestran en consola y no se guardan.")
        f = writer = None
    else:
        hechos = leer_anotados(out, solo_con_leds=args.salteados)
        pendientes = [fr for fr in frames_a_anotar(total, args.n) if fr not in hechos]
        print(f"Frames pendientes: {len(pendientes)}  ->  {out}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        nuevo = not os.path.exists(out)
        f = open(out, 'a', newline='')
        writer = csv.writer(f)
        if nuevo:
            writer.writerow(['frame', 'led', 'x', 'y'])

    try:
        # Lectura secuencial: el salto directo a un frame (CAP_PROP_POS_FRAMES) no es exacto en mp4
        n_frame = 0
        for objetivo in pendientes:
            while n_frame < objetivo:
                ok, frame = cap.read()
                if not ok:
                    break
                n_frame += 1
            if n_frame != objetivo:
                break
            res = anotar_frame(frame, n_frame)
            if res == 'quit':
                break
            if res == 'skip':
                res = [None] * LED_COUNT  # se registra como "ningun LED visible" para no volver a pedirlo
            if writer is None:
                marcas = ', '.join('faltante' if p is None else f"({p[0]:.2f}, {p[1]:.2f})" for p in res)
                print(f"Frame {n_frame} (no guardado): {marcas}")
                continue
            for i, p in enumerate(res):
                writer.writerow([n_frame, i, '' if p is None else f"{p[0]:.2f}", '' if p is None else f"{p[1]:.2f}"])
            f.flush()
            print(f"Frame {n_frame} guardado.")
    finally:
        if f is not None:
            f.close()
            compactar(out)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
