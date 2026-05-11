"""
Script para probar la deteccion de bordes y esquinas.
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Agregar la carpeta src al path para poder importar nuestra libreria
ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(ruta_src)

# Importar funciones de limpieza y extraccion de caracteristicas
from vision_lib.preprocessing import convertir_a_gris, transformar_intensidad, aplicar_filtro_espacial
from vision_lib.features import detectar_bordes, detectar_esquinas

# Paso 1: Cargar y limpiar la imagen original
imagen_bgr = cv2.imread('../tests/Tornillos.jpeg')
img_gris = convertir_a_gris(imagen_bgr)
img_contraste = transformar_intensidad(img_gris)
img_filtrada = aplicar_filtro_espacial(img_contraste, tipo="gaussiano")

# Paso 2: Detectar los contornos con Canny
bordes = detectar_bordes(img_filtrada, umbral_min=20, umbral_max=150)

# Paso 3: Encontrar esquinas con Harris
esquinas = detectar_esquinas(img_filtrada)

# Hacer una copia a color para dibujar los puntos detectados
img_esquinas_vis = imagen_bgr.copy()

# Pintar de rojo los pixeles que superen el umbral de sensibilidad (0.5%)
img_esquinas_vis[esquinas > 0.005 * esquinas.max()] = [0, 0, 255]

# Paso 4: Graficar todo junto para comparar
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
plt.title('1. Imagen Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(bordes, cmap='gray')
plt.title('2. Bordes (Canny)\n(Radiografia del contorno)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_esquinas_vis, cv2.COLOR_BGR2RGB))
plt.title('3. Esquinas (Harris)\n(Puntos en cambios bruscos)')
plt.axis('off')

plt.tight_layout()
plt.show()
