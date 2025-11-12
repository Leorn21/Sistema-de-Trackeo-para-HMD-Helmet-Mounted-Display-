# Sistema-de-Trackeo-para-HMD-Helmet-Mounted-Display-
En el presente proyecto se propone el desarrollo y ensayo de un software de procesamiento de imágenes que permita la detección robusta de marcadores ópticos y el cálculo preciso de sus centros ópticos. Dichos marcadores son tipo LEDs infrarrojos y desempeñan el papel de patrones ópticos de referencia

El objetivo final es crear un sistema que funcione en tiempo real bajo condiciones lumínicas variables, simulando un entorno complejo como la cabina de una aeronave.
¿Por qué es importante?
Para que un sistema de seguimiento (tracking) óptico funcione, se necesita saber con exactitud dónde están posicionados los marcadores de referencia. Esto se logra detectando los LEDs montados en una plataforma y calculando sus posiciones con la mayor precisión posible.
La precisión es crítica: un error en el cálculo del centro óptico, o la falla en detectar un LED por un cambio de luz, puede desalinear todo el sistema de seguimiento e invalidar los datos.

Herramientas Utilizadas:
Para desarrollar este sistema, se utilizó el siguiente stack de tecnologías:
•	Python 3.12.10: Lenguaje de programación principal.
•	OpenCV (cv2): Biblioteca de visión por computadora, esencial para el procesamiento de imágenes, conversión de espacios de color y algoritmos geométricos.
•	NumPy: Biblioteca para cálculos numéricos y la manipulación eficiente de las matrices de imagen.
•	Entorno Virtual (.venv): Para mantener un entorno de desarrollo aislado y gestionar las dependencias del proyecto.
