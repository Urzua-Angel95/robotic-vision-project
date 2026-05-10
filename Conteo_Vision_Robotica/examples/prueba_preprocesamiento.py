# -*- coding: utf-8 -*-
"""
Script de prueba para el modulo de preprocesamiento.
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt

ruta_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(ruta_src)

# Importamos TUS funciones creadas a mano
from vision_lib.preprocessing import convertir_a_gris, aplicar_filtro_espacial, transformar_intensidad

# 1. Cargar la imagen desde la carpeta tests
ruta_imagen = '../tests/Tornillos.jpeg'
imagen_bgr = cv2.imread(ruta_imagen)

if imagen_bgr is None:
    print("Error: No se pudo cargar la imagen. Verifica que esta en la carpeta 'tests'.")
else:
    # 2. PASO A PASO: Aplicar nuestro pipeline de preprocesamiento
    
    # A. Convertir a grises con nuestra formula matematica
    img_gris = convertir_a_gris(imagen_bgr)
    
    # B. Mejorar el contraste (estiramiento de intensidad)
    img_contraste = transformar_intensidad(img_gris)
    
    # C. Limpiar el ruido con nuestro filtro espacial (convolución matemática)
    img_filtrada = aplicar_filtro_espacial(img_contraste, tipo="gaussiano")
    
    # 3. GRAFICAR LOS RESULTADOS (Ideal para tu reporte técnico)
    # Convertimos la original de BGR a RGB solo para que Matplotlib la muestre con colores reales
    imagen_rgb_mostrar = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5)) 
    
    plt.subplot(1, 4, 1)
    plt.imshow(imagen_rgb_mostrar)
    plt.title('1. Original (RGB)')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_gris, cmap='gray')
    plt.title('2. Escala de Grises\n(Fórmula de Luminancia)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(img_contraste, cmap='gray')
    plt.title('3. Intensidad\n(Contraste Mejorado)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(img_filtrada, cmap='gray')
    plt.title('4. Filtro Espacial\n(Gaussiano sin ruido)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
