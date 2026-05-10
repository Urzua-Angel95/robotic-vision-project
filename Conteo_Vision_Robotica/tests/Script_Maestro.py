
"""
PROYECTO 2: Segmentación y Conteo de Objetos Industriales
Script Todo-en-Uno: Preprocesamiento, Segmentación, Conteo y Características.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
# 1. FUNCIONES DE PREPROCESAMIENTO
# =============================================================================
def convertir_a_gris(imagen_bgr):
    """Convierte a gris usando la matemática de luminancia."""
    B, G, R = imagen_bgr[:, :, 0], imagen_bgr[:, :, 1], imagen_bgr[:, :, 2]
    return ((0.299 * R) + (0.587 * G) + (0.114 * B)).astype(np.uint8)

def transformar_intensidad(imagen_gris):
    """Mejora el contraste expandiendo los valores de luz."""
    min_val, max_val = np.min(imagen_gris), np.max(imagen_gris)
    if max_val == min_val: return imagen_gris
    return ((imagen_gris - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)

def aplicar_filtro_espacial(imagen_gris):
    """Filtro Gaussiano matemático para limpiar el ruido."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    filtrada = signal.convolve2d(imagen_gris, kernel, mode='same', boundary='symm')
    return np.clip(filtrada, 0, 255).astype(np.uint8)

# =============================================================================
# 2. FUNCIONES DE SEGMENTACIÓN Y CONTEO
# =============================================================================
def umbralizar_imagen(imagen_gris):
    """Recorta el fondo usando el método automático de Otsu."""
    _, binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria

def segmentar_y_contar(imagen_binaria, area_minima=300):
    """Cuenta los objetos filtrando el polvo (Usaste 300 para arreglar la sombra)."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, 8)
    conteo = 0
    datos = []
    
    for i in range(1, num_labels): # Ignoramos el 0 que es el fondo negro
        x, y, w, h, area = stats[i]
        if area > area_minima:
            conteo += 1
            datos.append({"id": conteo, "x": x, "y": y, "w": w, "h": h})
            
    return conteo, datos

# =============================================================================
# 3. FUNCIONES DE CARACTERÍSTICAS GEOMÉTRICAS
# =============================================================================
def detectar_bordes(imagen_gris):
    """Radiografía del contorno con Canny."""
    return cv2.Canny(imagen_gris, 20, 150)

def detectar_esquinas(imagen_gris):
    """Detección de puntos de quiebre y roscas con Harris."""
    gray_float = np.float32(imagen_gris)
    esquinas = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.002)
    return cv2.dilate(esquinas, None)

# =============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN (PIPELINE)
# =============================================================================
if __name__ == "__main__":
    # 1. Cargar la imagen (Asegúrate de que se llame así y esté en la misma carpeta)
    ruta_imagen = 'Semillas.png'
    img_color = cv2.imread(ruta_imagen)
    
    if img_color is None:
        print(f"Error: No se encontró '{ruta_imagen}'.")
    else:
        # Copias para dibujar
        img_resultados = img_color.copy()
        img_esquinas_vis = img_color.copy()
        
        # 2. Aplicar Pipeline de Preprocesamiento
        img_gris = convertir_a_gris(img_color)
        img_contraste = transformar_intensidad(img_gris)
        img_limpia = aplicar_filtro_espacial(img_contraste)
        
        # 3. Aplicar Segmentación y Conteo (Área mínima 300 como descubriste)
        img_binaria = umbralizar_imagen(img_limpia)
        total_tornillos, datos_tornillos = segmentar_y_contar(img_binaria, area_minima=300)
        
        # Dibujar rectángulos del conteo
        for obj in datos_tornillos:
            cv2.rectangle(img_resultados, (obj["x"], obj["y"]), 
                          (obj["x"] + obj["w"], obj["y"] + obj["h"]), (0, 255, 0), 3)
            cv2.putText(img_resultados, f"#{obj['id']}", (obj["x"], obj["y"]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
        # 4. Extraer Características (Bordes y Esquinas)
        bordes = detectar_bordes(img_limpia)
        esquinas = detectar_esquinas(img_limpia)
        # Marcamos las esquinas fuertes en rojo (al 5% para que no se llene todo el tornillo)
        img_esquinas_vis[esquinas > 0.05 * esquinas.max()] = [0, 0, 255]
        
        # 5. Mostrar TODO el proceso al profesor en una sola ventana
        print(f"--- ANÁLISIS COMPLETADO ---")
        print(f"Total de tornillos detectados: {total_tornillos}")
        
        plt.figure(figsize=(16, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title('1. Original')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(img_limpia, cmap='gray')
        plt.title('2. Preprocesamiento (Gris+Contraste+Filtro)')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(img_binaria, cmap='gray')
        plt.title('3. Segmentación (Otsu Binario)')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(img_resultados, cv2.COLOR_BGR2RGB))
        plt.title(f'4. Conteo Final: {total_tornillos}')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(bordes, cmap='gray')
        plt.title('5. Bordes (Canny)')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(img_esquinas_vis, cv2.COLOR_BGR2RGB))
        plt.title('6. Esquinas (Harris)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
