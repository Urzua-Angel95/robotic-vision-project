"""
Script para probar la segmentacion y el conteo de objetos.
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Agregar la carpeta src al path
ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(ruta_src)

from vision_lib.preprocessing import convertir_a_gris, transformar_intensidad, aplicar_filtro_espacial
from vision_lib.segmentation import umbralizar_imagen, segmentar_y_contar

# Paso 1: Cargar la imagen
imagen_bgr = cv2.imread('../tests/Semillas.png')
imagen_mostrar = imagen_bgr.copy()

# Paso 2: Preprocesamiento basico
img_gris = convertir_a_gris(imagen_bgr)
img_contraste = transformar_intensidad(img_gris)
img_filtrada = aplicar_filtro_espacial(img_contraste, tipo="gaussiano")

# Paso 3: Binarizacion y conteo
# Usamos Otsu para encontrar el umbral ideal automaticamente
img_binaria = umbralizar_imagen(img_filtrada, metodo="otsu")

# Contamos los objetos y filtramos los que tengan menos de 300 pixeles de area
total_tornillos, datos_tornillos = segmentar_y_contar(img_binaria, area_minima=300)

print("\n--- RESULTADOS DEL CONTEO ---")
print(f"Total de objetos detectados: {total_tornillos}\n")

# Paso 4: Dibujar los resultados en la imagen
for obj in datos_tornillos:
    x, y = obj["x"], obj["y"]
    w, h = obj["ancho"], obj["alto"]
    
    # Dibujar un rectangulo verde
    cv2.rectangle(imagen_mostrar, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # Poner el ID del objeto
    cv2.putText(imagen_mostrar, f"#{obj['id']}", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Mostrar las graficas
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_binaria, cmap='gray')
plt.title('Imagen Binaria (Fondo separado)')
plt.axis('off')

plt.subplot(1, 2, 2)
# Convertir a RGB para que matplotlib muestre bien los colores
plt.imshow(cv2.cvtColor(imagen_mostrar, cv2.COLOR_BGR2RGB))
plt.title(f'Conteo Final: {total_tornillos} tornillos')
plt.axis('off')

plt.tight_layout()
plt.show()