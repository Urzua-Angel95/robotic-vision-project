# -*- coding: utf-8 -*-
"""
Script de prueba para la detección de bordes y esquinas.
"""
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(ruta_src)

# Importamos funciones de limpieza y características
from vision_lib.preprocessing import convertir_a_gris, transformar_intensidad, aplicar_filtro_espacial
from vision_lib.features import detectar_bordes, detectar_esquinas

# 1. Cargar y limpiar la imagen
imagen_bgr = cv2.imread('../tests/Tornillos.jpeg')
img_gris = convertir_a_gris(imagen_bgr)
img_contraste = transformar_intensidad(img_gris)
img_filtrada = aplicar_filtro_espacial(img_contraste, tipo="gaussiano")

# 2. NUEVO: Detección de Bordes (Canny)
bordes = detectar_bordes(img_filtrada, umbral_min=20, umbral_max=150)

# 3. NUEVO: Detección de Esquinas (Harris)
esquinas = detectar_esquinas(img_filtrada)

# Preparamos una imagen a color para pintar los puntos de las esquinas en rojo
img_esquinas_vis = imagen_bgr.copy()
# Marcamos en rojo [0, 0, 255] los píxeles donde Harris detectó una esquina fuerte
img_esquinas_vis[esquinas > 0.005 * esquinas.max()] = [0, 0, 255]

# 4. Mostrar resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
plt.title('1. Imagen Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(bordes, cmap='gray')
plt.title('2. Bordes (Canny)\n(Radiografía del contorno)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_esquinas_vis, cv2.COLOR_BGR2RGB))
plt.title('3. Esquinas (Harris)\n(Puntos rojos en cambios bruscos)')
plt.axis('off')

plt.tight_layout()
plt.show()

