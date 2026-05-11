"""
Modulo de limpieza y preparacion de imagenes.
"""
import numpy as np
from scipy import signal

def convertir_a_gris(imagen_bgr):
    """
    Pasa la imagen a blanco y negro usando la formula de luminancia.
    """
    # Extraemos los tres colores por separado
    B = imagen_bgr[:, :, 0]
    G = imagen_bgr[:, :, 1]
    R = imagen_bgr[:, :, 2]
    
    # Aplicamos la formula basada en la vista humana
    gris = (0.299 * R) + (0.587 * G) + (0.114 * B)
    
    # Aseguramos que quede como enteros de 8 bits 
    return gris.astype(np.uint8)


def aplicar_filtro_espacial(imagen_gris, tipo="gaussiano"):
    """
    Limpia el ruido de la imagen usando una convolucion matematica.
    """
    if tipo == "gaussiano":
        # Matriz gaussiana 3x3 para suavizar cuidando los bordes
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16.0
                           
    elif tipo == "promedio":
        # Matriz simple que promedia todo por igual
        kernel = np.ones((3, 3)) / 9.0
        
    else:
        raise ValueError("Opcion no valida. Intenta con 'gaussiano' o 'promedio'.")
        
    # Aplicar el filtro a la imagen completa
    imagen_filtrada = signal.convolve2d(imagen_gris, kernel, mode='same', boundary='symm')
    
    # Evitamos que los valores se salgan del rango de luz permitido
    return np.clip(imagen_filtrada, 0, 255).astype(np.uint8)


def transformar_intensidad(imagen_gris):
    """
    Mejora el contraste estirando los valores de luz al maximo.
    """
    # Encontramos el pixel mas oscuro y el mas claro de la foto
    min_val = np.min(imagen_gris)
    max_val = np.max(imagen_gris)
    
    # Por si la imagen es de un solo color y evitar dividir entre 0
    if max_val == min_val:
        return imagen_gris
        
    # Normalizacion Min-Max para estirar el contraste
    imagen_estirada = (imagen_gris - min_val) * (255.0 / (max_val - min_val))
    
    return imagen_estirada.astype(np.uint8)