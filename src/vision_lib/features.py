"""
Modulo de Extraccion de Caracteristicas
Contiene funciones para detectar bordes y esquinas matematicas en los objetos.
"""

import cv2
import numpy as np

def detectar_bordes(imagen_gris, umbral_min=20, umbral_max=150):
    """
    Detecta los bordes de los objetos en una imagen usando el algoritmo de Canny.
    """
    # El algoritmo de Canny calcula los gradientes de luz para encontrar contornos
    bordes = cv2.Canny(imagen_gris, umbral_min, umbral_max)
    
    return bordes


def detectar_esquinas(imagen_gris, sensibilidad=0.001):
    """
    Detecta las esquinas en una imagen usando el detector de Harris.
    """
    # El algoritmo de Harris requiere que la imagen este en formato de punto flotante
    gray_float = np.float32(imagen_gris)
    
    # blockSize=2 (tamano de vecindario), ksize=3 (apertura de Sobel)
    esquinas = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=sensibilidad)
    
    # Dilatamos un poco los puntos resultantes solo para que sean mas faciles de ver
    esquinas = cv2.dilate(esquinas, None)
    
    return esquinas
