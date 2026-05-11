"""
Script para probar el preprocesamiento de la imagen.
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Agregar la carpeta src al path
ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(ruta_src)

# Importar nuestras propias funciones
from vision_lib.preprocessing import convertir_a_gris, aplicar_filtro_espacial, transformar_intensidad

# Paso 1: Cargar la imagen de prueba
ruta_imagen = '../tests/Semillas.png'
imagen_bgr = cv2.imread(ruta_imagen)

if imagen_bgr is None:
    print("Error: No se pudo cargar la imagen. Revisa la ruta.")
else:
    # Paso 2: Aplicar los filtros uno por uno
    
    # A. Pasar a grises
    img_gris = convertir_a_gris(imagen_bgr)
    
    # B. Estirar el contraste para que resalte mas
    img_contraste = transformar_intensidad(img_gris)
    
    # C. Suavizar con filtro gaussiano para quitar ruido
    img_filtrada = aplicar_filtro_espacial(img_contraste, tipo="gaussiano")
    
    # Paso 3: Mostrar el antes y el despues
    # OpenCV usa BGR, asi que lo pasamos a RGB para que matplotlib lo muestre bien
    imagen_rgb_mostrar = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5)) 
    
    plt.subplot(1, 4, 1)
    plt.imshow(imagen_rgb_mostrar)
    plt.title('1. Original (RGB)')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_gris, cmap='gray')
    plt.title('2. Escala de Grises\n(Luminancia)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(img_contraste, cmap='gray')
    plt.title('3. Intensidad\n(Mejor contraste)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(img_filtrada, cmap='gray')
    plt.title('4. Filtro Espacial\n(Gaussiano)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
