"""
Módulo de Segmentación
Contiene funciones para separar el fondo de los objetos y realizar el conteo de regiones.
"""

import cv2
import numpy as np

def umbralizar_imagen(imagen_gris, metodo="otsu", umbral_manual=127):
    """
    Convierte una imagen en escala de grises a una imagen binaria (blanco y negro puro).
    
    Parámetros:
        imagen_gris (numpy.ndarray): Imagen preprocesada en escala de grises.
        metodo (str): 'manual' para usar un valor fijo, 'otsu' para cálculo automático.
        umbral_manual (int): Valor de corte si se usa el método manual (0-255).
        
    Retorna:
        numpy.ndarray: Imagen binaria donde los objetos son blancos (255) y el fondo negro (0).
    """
    if metodo == "manual":
        # Implementación matemática propia usando numpy
        # Todo píxel mayor al umbral se vuelve 255, el resto 0
        binaria = np.where(imagen_gris > umbral_manual, 255, 0).astype(np.uint8)
        return binaria
        
    elif metodo == "otsu":
        # El método de Otsu de OpenCV calcula el "valle" ideal en el histograma automáticamente
        umbral_calculado, binaria = cv2.threshold(
            imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binaria
        
    else:
        raise ValueError("Método no soportado. Usa 'manual' u 'otsu'.")


def segmentar_y_contar(imagen_binaria, area_minima=200):
    """
    Encuentra las regiones conectadas (objetos) en la imagen binaria y las cuenta,
    filtrando el ruido por tamaño.
    
    Parámetros:
        imagen_binaria (numpy.ndarray): Imagen de entrada con fondo negro y objetos blancos.
        area_minima (int): Cantidad mínima de píxeles para que una región se cuente como objeto real.
        
    Retorna:
        int: Conteo total de objetos válidos.
        list: Lista de diccionarios con los datos (coordenadas y área) de cada objeto.
    """
    # Algoritmo de conectividad de 8 vecinos para agrupar píxeles
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, 8)
    
    conteo_final = 0
    datos_objetos = []
    
    # Empezamos en 1 porque el índice 0 siempre es el fondo negro
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Filtro de seguridad: ignoramos basurita o ruido minúsculo
        if area > area_minima:
            conteo_final += 1
            
            # Guardamos las características básicas que pide tu rúbrica
            datos_objetos.append({
                "id": conteo_final,
                "x": x,
                "y": y,
                "ancho": w,
                "alto": h,
                "area": area,
                "centro_x": int(centroids[i][0]),
                "centro_y": int(centroids[i][1])
            })
            
    return conteo_final, datos_objetos