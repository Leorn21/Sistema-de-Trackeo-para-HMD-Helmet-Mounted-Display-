"""Error de localizacion de los LEDs contra la anotacion manual (verdad de terreno).

Uso:
    python evaluar_error.py --video ../Videos/Video_LEDs_nuevo.mp4
    python evaluar_error.py --repetibilidad anotaciones/Video_LEDs_nuevo_b.csv   (error de la propia anotacion)

Para cada LED anotado se mide la distancia euclidea (px) entre el centro anotado y:
  - la DETECCION: el centroide medido en ese frame (sin filtrar),
  - el KALMAN: la posicion estimada por el filtro, que es la salida final del sistema.
Tambien se expresa relativa al espaciado entre LEDs, que no depende de la distancia a la camara.
"""
import argparse
import csv
import os

import numpy as np

import Deteccion_leds as D

GROSS_ERROR = 0.5  # en espaciados: a mas de medio espaciado, el LED se asigno a otro (error de identidad)


def leer_anotaciones(path):
    gt = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['x'] and r['y']:
                gt[(int(r['frame']), int(r['led']))] = np.array([float(r['x']), float(r['y'])])
    return gt


def estadisticas(e):
    e = np.asarray(e, dtype=float)
    if len(e) == 0:
        return None
    return {'n': len(e), 'media': e.mean(), 'desvio': e.std(ddof=1) if len(e) > 1 else 0.0,
            'mediana': np.median(e), 'rmse': np.sqrt((e**2).mean()),
            'p95': np.percentile(e, 95), 'max': e.max()}


def imprimir_fila(nombre, st, unidad=''):
    if st is None:
        print(f"  {nombre:<34} sin muestras")
        return
    print(f"  {nombre:<34} n={st['n']:<4} media={st['media']:.2f}{unidad}  desvio={st['desvio']:.2f}  "
          f"mediana={st['mediana']:.2f}  RMSE={st['rmse']:.2f}  P95={st['p95']:.2f}  max={st['max']:.2f}")


def evaluar(video, anotaciones, out_csv, model, sigma_accel, sigma_jerk):
    gt = leer_anotaciones(anotaciones)
    frames = {f for f, _ in gt}
    registros = []

    def on_frame(n, frame, tracker, candidates, matched):
        if n not in frames:
            return
        for (f, i), p_gt in gt.items():
            if f != n:
                continue
            r = {'frame': n, 'led': i, 'x_gt': p_gt[0], 'y_gt': p_gt[1],
                 'espaciado': tracker.spacing, 'vel_px_frame': np.nan,
                 'err_det': np.nan, 'err_kalman': np.nan}
            if tracker.initialized:
                kf = tracker.filters[i]
                r['err_kalman'] = float(np.linalg.norm(kf.position - p_gt))
                r['vel_px_frame'] = float(np.linalg.norm(kf.kalman.statePost[2:4, 0]) * tracker.dt)
                if i in matched:
                    r['err_det'] = float(np.linalg.norm(np.array(candidates[matched[i]]['center']) - p_gt))
            registros.append(r)

    D.procesar_video(video, model, sigma_accel, sigma_jerk, display=False, verbose=False, on_frame=on_frame)

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(registros[0].keys()) if registros else ['frame'])
        w.writeheader()
        w.writerows(registros)

    total = len(registros)
    trackeados = [r for r in registros if not np.isnan(r['err_kalman'])]
    print(f"\n=== Error de localizacion: {os.path.basename(video)} ({len(frames)} frames, {total} LEDs anotados) ===")
    print(f"LEDs con tracking activo: {len(trackeados)}/{total} ({100 * len(trackeados) / max(total, 1):.1f}%)")
    det = [r for r in trackeados if not np.isnan(r['err_det'])]
    print(f"LEDs detectados en el frame: {len(det)}/{total} ({100 * len(det) / max(total, 1):.1f}%)")
    if not trackeados:
        return

    grueso = [r for r in trackeados if r['err_kalman'] > GROSS_ERROR * r['espaciado']]
    print(f"Errores de identidad (> {GROSS_ERROR} espaciado): {len(grueso)}")
    validos = [r for r in trackeados if r not in grueso]
    esp = np.mean([r['espaciado'] for r in trackeados])
    print(f"Espaciado medio entre LEDs: {esp:.1f} px\n")

    print("Error en pixeles (sin errores de identidad):")
    imprimir_fila("Deteccion (centroide medido)", estadisticas([r['err_det'] for r in validos if not np.isnan(r['err_det'])]), ' px')
    imprimir_fila("Kalman (salida del sistema)", estadisticas([r['err_kalman'] for r in validos]), ' px')
    print("\nError relativo al espaciado entre LEDs (%):")
    imprimir_fila("Deteccion", estadisticas([100 * r['err_det'] / r['espaciado'] for r in validos if not np.isnan(r['err_det'])]), ' %')
    imprimir_fila("Kalman", estadisticas([100 * r['err_kalman'] / r['espaciado'] for r in validos]), ' %')

    # Como cambia el error con la velocidad: clave para extrapolar al brazo robotico
    vel = np.array([r['vel_px_frame'] for r in validos])
    cortes = np.percentile(vel, [33.3, 66.7])
    print(f"\nError del Kalman segun velocidad de la barra (px/frame):")
    for nombre, sel in [(f"lenta  (< {cortes[0]:.1f})", vel < cortes[0]),
                        (f"media  ({cortes[0]:.1f} - {cortes[1]:.1f})", (vel >= cortes[0]) & (vel < cortes[1])),
                        (f"rapida (>= {cortes[1]:.1f})", vel >= cortes[1])]:
        imprimir_fila(nombre, estadisticas([r['err_kalman'] for r, s in zip(validos, sel) if s]), ' px')
    print(f"\nDetalle por LED guardado en: {out_csv}")


def repetibilidad(a, b):
    """Error entre dos anotaciones del mismo video: el piso de precision de la propia medicion."""
    ga, gb = leer_anotaciones(a), leer_anotaciones(b)
    comunes = sorted(set(ga) & set(gb))
    print(f"\n=== Repetibilidad de la anotacion ({len(comunes)} LEDs en comun) ===")
    imprimir_fila("Diferencia entre anotaciones", estadisticas([np.linalg.norm(ga[k] - gb[k]) for k in comunes]), ' px')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Error de localizacion contra anotacion manual")
    parser.add_argument('--video', default=D.DEFAULT_VIDEO)
    parser.add_argument('--anotaciones', help="CSV de anotar_leds.py (por defecto anotaciones/<video>.csv)")
    parser.add_argument('--repetibilidad', help="Segundo CSV del mismo video para medir el error de anotacion")
    parser.add_argument('--model', choices=['cv', 'ca'], default=D.KALMAN_MODEL)
    parser.add_argument('--sigma-accel', type=float, default=D.SIGMA_ACCEL)
    parser.add_argument('--sigma-jerk', type=float, default=D.SIGMA_JERK)
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anotaciones',
                        os.path.splitext(os.path.basename(args.video))[0])
    anot = args.anotaciones or base + '.csv'
    if args.repetibilidad:
        repetibilidad(anot, args.repetibilidad)
    else:
        evaluar(args.video, anot, os.path.splitext(anot)[0] + '_errores.csv', args.model, args.sigma_accel, args.sigma_jerk)
