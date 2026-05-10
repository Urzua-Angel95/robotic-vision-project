from setuptools import setup, find_packages

setup(
   
    name="vision_lib",
   
    version="0.1.0",
    
 
    description="Libreria para segmentacion, conteo y analisis de objetos industriales.",
    
    
    author="Equipo10_Rodriguez_Carreon_Urzua_Hernandez",
    
   
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
   
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
        "matplotlib"
    ],
    
    python_requires=">=3.7",
)
