"""
Modulo de Segmentacion
Contiene funciones para separar el fondo y contar los objetos.
"""

import cv2
import numpy as np

def umbralizar_imagen(imagen_gris, metodo="otsu", umbral_manual=127):
    """
    Convierte la imagen a blanco y negro puro para separar los objetos del fondo.
    """
    if metodo == "manual":
        # Todo pixel mayor al umbral se vuelve blanco (255), el resto negro (0)
        binaria = np.where(imagen_gris > umbral_manual, 255, 0).astype(np.uint8)
        return binaria
        
    elif metodo == "otsu":
        # Otsu busca el punto de corte ideal automaticamente basandose en el histograma
        umbral_calculado, binaria = cv2.threshold(
            imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binaria
        
    else:
        raise ValueError("Metodo no valido. Usa 'manual' u 'otsu'.")


def segmentar_y_contar(imagen_binaria, area_minima=200):
    """
    Encuentra los objetos en la imagen binaria, los cuenta y filtra el ruido.
    """
    # Agrupamos pixeles vecinos para formar los objetos completos
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, 8)
    
    conteo_final = 0
    datos_objetos = []
    
    # Empezamos en 1 para ignorar el fondo negro (que siempre es el indice 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Ignoramos cualquier manchita que sea mas pequena que el area minima
        if area > area_minima:
            conteo_final += 1
            
            # Guardamos la info basica del objeto
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