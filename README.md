**Sistema de Trackeo para HMD (Helmet Mounted Display)**

----Descripción General----

Este proyecto consiste en el desarrollo y validación de un software de visión artificial diseñado para la detección robusta de marcadores ópticos tipo LED. El sistema está optimizado para funcionar en condiciones lumínicas variables, simulando los desafíos presentes en la cabina de una aeronave, con el fin de proporcionar un seguimiento preciso para dispositivos de visualización montados en casco (HMD).

----Objetivo del Proyecto----

Desarrollar un algoritmo capaz de:

Detección Robusta: Identificar marcadores LED en entornos con reflejos, deslumbramientos y sombras.

Precisión Sub-píxel: Calcular los centros ópticos de los marcadores con alta fidelidad mediante métodos geométricos.

Simulación de Entorno: Validar el rendimiento en escenarios que replican la complejidad lumínica de una cabina aeronáutica.

-----Metodología Técnica-----

El sistema emplea técnicas avanzadas de procesamiento de imágenes con OpenCV y Python:

Preprocesamiento en Espacio HSV: Conversión de imágenes para aislar el matiz (Hue) y la saturación (Saturation), permitiendo una segmentación de color resistente a cambios de intensidad lumínica.

Segmentación por Umbralización: Aplicación de máscaras binarias para aislar los puntos de interés (LEDs púrpura/violeta).

Filtrado Morfológico: Uso de operaciones de apertura y cierre para eliminar ruido y consolidar la forma de los marcadores.

Ajuste de Elipses (Robustez de Perspectiva): Implementación de cv2.fitEllipse para calcular el centroide de los LEDs, lo que permite mantener la precisión incluso cuando la forma circular se deforma por el ángulo de la cámara.

-----Requisitos-----

OpenCV (opencv-python)
NumPy

-----Instalación y Uso-----

1)Clonar el repositorio:
git clone https://github.com/Leorn21/Sistema-de-Trackeo-para-HMD-Helmet-Mounted-Display-.git

2)Instalar dependencias:
pip install opencv-python numpy

3)Ejecutar el script principal:
python Deteccion_leds.py
