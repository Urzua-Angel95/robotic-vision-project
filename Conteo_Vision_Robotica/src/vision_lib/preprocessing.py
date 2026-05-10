"""
Modulo de Preprocesamiento
"""
import numpy as np
from scipy import signal

def convertir_a_gris(imagen_bgr):
    """
    Convierte una imagen a color a escala de grises.
    Implementación matemática propia usando la ecuación de luminancia.
    
    Parámetros:
        imagen_bgr (numpy.ndarray): Matriz de la imagen a color (Alto, Ancho, 3).
                                    Asume formato BGR (estándar de OpenCV).
    Retorna:
        numpy.ndarray: Imagen en escala de grises (Alto, Ancho).
    """
    # Separamos los canales de color
    B = imagen_bgr[:, :, 0]
    G = imagen_bgr[:, :, 1]
    R = imagen_bgr[:, :, 2]
    
    # Ecuación de luminancia perceptiva (basada en cómo ve el ojo humano)
    gris = (0.299 * R) + (0.587 * G) + (0.114 * B)
    
    # Aseguramos que el resultado esté en el formato correcto (enteros de 0 a 255)
    return gris.astype(np.uint8)


def aplicar_filtro_espacial(imagen_gris, tipo="gaussiano"):
    """
    Aplica un filtro espacial mediante convolución 2D para reducir el ruido.
    
    Parámetros:
        imagen_gris (numpy.ndarray): Imagen en 2D.
        tipo (str): 'gaussiano' para preservar bordes o 'promedio' para difuminado general.
        
    Retorna:
        numpy.ndarray: Imagen filtrada.
    """
    if tipo == "gaussiano":
        # Kernel de suavizado gaussiano 3x3 (le da más peso al píxel central)
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16.0
                           
    elif tipo == "promedio":
        # Kernel de promedio simple 3x3 (todos los pesos valen lo mismo)
        kernel = np.ones((3, 3)) / 9.0
        
    else:
        raise ValueError("Tipo de filtro no soportado. Usa 'gaussiano' o 'promedio'.")
        
    # Convolución matemática 2D usando scipy
    # mode='same' asegura que la imagen resultante tenga el mismo tamaño
    # boundary='symm' maneja los bordes de la imagen para que no salgan oscuros
    imagen_filtrada = signal.convolve2d(imagen_gris, kernel, mode='same', boundary='symm')
    
    # Recortamos valores para que no se salgan del rango 0-255 por el cálculo matemático
    return np.clip(imagen_filtrada, 0, 255).astype(np.uint8)


def transformar_intensidad(imagen_gris):
    """
    Aplica una transformación de intensidad (Estiramiento de contraste).
    Expande matemáticamente los valores de luz para mejorar imágenes opacas.
    
    Parámetros:
        imagen_gris (numpy.ndarray): Imagen en 2D a transformar.
        
    Retorna:
        numpy.ndarray: Imagen con contraste mejorado.
    """
    # Buscamos el píxel más oscuro y el más claro de la imagen real
    min_val = np.min(imagen_gris)
    max_val = np.max(imagen_gris)
    
    # Prevención de error por división entre cero (si la imagen es de un solo color sólido)
    if max_val == min_val:
        return imagen_gris
        
    # Fórmula matemática de Normalización Min-Max
    imagen_estirada = (imagen_gris - min_val) * (255.0 / (max_val - min_val))
    
    return imagen_estirada.astype(np.uint8)

