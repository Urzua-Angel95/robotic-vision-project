# Vision Industrial Lib: Segmentacion y Conteo Automatico

Un paquete de Python disenado para el procesamiento de imagenes industriales, segmentacion de objetos y extraccion de caracteristicas geometricas. Este proyecto fue construido desde cero implementando modelos matematicos propios (como luminancia y filtros espaciales) para garantizar un control total sobre el pipeline de vision artificial.

## Caracteristicas Principales

Este paquete esta dividido en tres modulos principales:

1. **Preprocesamiento (`preprocessing.py`):**
   * Conversion a escala de grises mediante la ecuacion de Luminancia Perceptiva.
   * Mejora de contraste usando Normalizacion Min-Max.
   * Reduccion de ruido con Filtrado Espacial Gaussiano.

2. **Segmentacion (`segmentation.py`):**
   * Separacion de fondo y objeto utilizando umbralizacion automatica de Otsu.
   * Reparacion de siluetas fragmentadas por sombras o brillos mediante Cierre Morfologico.
   * Conteo automatico de piezas mediante Etiquetado de Componentes Conectados (con filtrado inteligente de ruido por area minima).

3. **Extraccion de Caracteristicas (`features.py`):**
   * Deteccion de siluetas exactas (Canny).
   * Deteccion de esquinas y texturas, ideal para roscas de tornillos (Harris).

```Requisitos e Instalacion 
Este paquete requiere Python 3.7 o superior. Las dependencias principales (numpy, scipy, opencv-python, matplotlib) se instalaran automaticamente.

Para instalar la libreria de forma local, abre tu terminal, navega a la carpeta raiz del proyecto y ejecuta:

pip install .

Si deseas instalarlo en "modo editable" (para poder modificar el codigo fuente sin tener que reinstalar):

pip install -e .

Una vez instalada la libreria, puedes importar las funciones en cualquier script de Python de la siguiente manera:

from vision_lib.preprocessing import convertir_a_gris, transformar_intensidad, aplicar_filtro_espacial
from vision_lib.segmentation import umbralizar_imagen, segmentar_y_contar