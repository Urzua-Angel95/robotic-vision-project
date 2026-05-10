# -*- coding: utf-8 -*-
"""
Script de prueba para la segmentación y el conteo.
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Conectamos con tu librería
ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(ruta_src)

from vision_lib.preprocessing import convertir_a_gris, transformar_intensidad, aplicar_filtro_espacial
from vision_lib.segmentation import umbralizar_imagen, segmentar_y_contar

# 1. Cargar la imagen
imagen_bgr = cv2.imread('../tests/Tornillos.jpeg')
imagen_mostrar = imagen_bgr.copy()

# 2. Reutilizar el preprocesamiento del paso anterior
img_gris = convertir_a_gris(imagen_bgr)
img_contraste = transformar_intensidad(img_gris)
img_filtrada = aplicar_filtro_espacial(img_contraste, tipo="gaussiano")

# 3. NUEVO: Umbralización y Conteo
# Usamos Otsu para que detecte el umbral ideal automáticamente
img_binaria = umbralizar_imagen(img_filtrada, metodo="otsu")

# Contamos los objetos (filtramos todo lo menor a 200 píxeles de área)
total_tornillos, datos_tornillos = segmentar_y_contar(img_binaria, area_minima=300)

print("\n--- RESULTADOS DEL CONTEO ---")
print(f"\n Total de objetos detectados: {total_tornillos}")

# 4. Dibujar rectángulos alrededor de los tornillos detectados
for obj in datos_tornillos:
    x, y = obj["x"], obj["y"]
    w, h = obj["ancho"], obj["alto"]
    
    # Dibujamos una caja verde
    cv2.rectangle(imagen_mostrar, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # Escribimos el número del objeto
    cv2.putText(imagen_mostrar, f"#{obj['id']}", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Mostrar el resultado visual
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_binaria, cmap='gray')
plt.title('Imagen Binaria (Fondo separado)')
plt.axis('off')

plt.subplot(1, 2, 2)
# Convertir a RGB para matplotlib
plt.imshow(cv2.cvtColor(imagen_mostrar, cv2.COLOR_BGR2RGB))
plt.title(f'Conteo Final: {total_tornillos} tornillos')
plt.axis('off')

plt.tight_layout()
plt.show()