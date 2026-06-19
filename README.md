# Directorio de Datos - Clasificador de Residuos

Este directorio contiene la estructura y las particiones correspondientes al dataset de 9 clases (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash) utilizado para el proyecto.

## Origen y Descarga

El dataset fue consolidado y limpiado manualmente de imagenes de tipo catalogo. Por restricciones de tamaño de GitHub, las imágenes binarias se encuentran excluidas del control de versiones mediante el archivo `.gitignore`.

La descarga automática y extracción de la estructura del dataset se realiza directamente desde la primera celda del notebook experimental alojado en la carpeta `dev/`.

## Link del dataset, alojado públicamente en la plataforma Kaggle.

https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset

## Antes de clonar: Abrir consola y ejecutar: 
python -m pip install pandas numpy torch torchvision scikit-learn matplotlib seaborn pillow tqdm

python -m pip install gdown

## Luego de instalar dependencias:
git clone url


## Luego de clonar proyecto:
Abrir consola, navegar hasta raíz del proyecto y ejecutar: python dev/proyecto.py
