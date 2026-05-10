"""
Módulo de Extracción de Características
Contiene funciones para detectar bordes y esquinas matemáticas en los objetos.
"""

import cv2
import numpy as np

def detectar_bordes(imagen_gris, umbral_min=20, umbral_max=150):
    """
    Detecta los bordes de los objetos en una imagen usando el algoritmo de Canny.
    
    Parámetros:
        imagen_gris (numpy.ndarray): Imagen en escala de grises (preferiblemente suavizada).
        umbral_min (int): Límite inferior para detectar bordes débiles.
        umbral_max (int): Límite superior para detectar bordes fuertes.
        
    Retorna:
        numpy.ndarray: Imagen binaria con los bordes delineados.
    """
    # El algoritmo de Canny calcula los gradientes de luz para encontrar contornos
    bordes = cv2.Canny(imagen_gris, umbral_min, umbral_max)
    return bordes


def detectar_esquinas(imagen_gris, sensibilidad=0.001):
    """
    Detecta las esquinas en una imagen usando el detector de Harris.
    
    Parámetros:
        imagen_gris (numpy.ndarray): Imagen en escala de grises.
        sensibilidad (float): Parámetro 'k' de Harris. Valores más bajos detectan más esquinas.
        
    Retorna:
        numpy.ndarray: Mapa de respuesta de esquinas.
    """
    # El algoritmo de Harris requiere que la imagen esté en formato de punto flotante
    gray_float = np.float32(imagen_gris)
    
    # blockSize=2 (tamaño de vecindario), ksize=3 (apertura de Sobel)
    esquinas = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=sensibilidad)
    
    # Dilatamos un poco los puntos resultantes solo para que sean más fáciles de ver
    esquinas = cv2.dilate(esquinas, None)
    
    return esquinas

