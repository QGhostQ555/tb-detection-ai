# Detección Automatizada de Tuberculosis en Imágenes de Rayos-X Torácico Usando Fine-Tuning de Redes Profundas Preentrenadas

Proyecto de clasificación binaria (`TB` vs `NORMAL`) usando transfer learning con EfficientNet-B4.

## Tecnologías Utilizadas

Este proyecto está desarrollado en **Python** y usa un pipeline de **Deep Learning + procesamiento de imágenes médicas** para clasificar radiografías de tórax en dos clases: `NORMAL` y `TB`.

### Stack principal

- **Python 3.10**
- **PyTorch (`torch`)**: framework principal para entrenamiento e inferencia.
- **Torchvision (`torchvision`)**: modelos preentrenados y transformaciones de imagen.
- **EfficientNet-B4**: arquitectura base de clasificación (Transfer Learning).
- **Torch-DirectML (`torch-directml`)**: aceleración en GPU AMD.
- **NumPy**: operaciones numéricas.
- **Scikit-learn (`scikit-learn`)**: métricas de evaluación (AUC, F1, matriz de confusión, etc.).
- **OpenCV (`opencv-python`)**: técnicas de mejora de imagen para radiografías.
- **PIL/Pillow**: lectura y manipulación de imágenes en el pipeline.
- **JSON**: almacenamiento de métricas y resultados del entrenamiento.

## Tecnología Aplicada a Imágenes (Radiografías CXR)

El proyecto incluye un flujo específico para imágenes médicas de tórax:

- Redimensionamiento de imagen a resolución configurable (`--img-size`).
- Conversión de imagen a escala de grises y posterior adaptación a 3 canales (RGB) para modelos preentrenados.
- Normalización con estadísticas de ImageNet para compatibilidad con EfficientNet-B4.
- Aumentación de datos (data augmentation) controlada:
  - rotación leve (`--rotation-deg`)
  - flip horizontal configurable (`--hflip-prob`)
  - ajustes suaves de brillo/contraste para robustez

### Técnicas de enhancement disponibles

Se pueden activar desde `train.py` con `--enhancement-mode`:

- `none`: sin mejora adicional.
- `clahe`: mejora de contraste local con CLAHE.
- `clahe_gamma`: CLAHE + corrección gamma.
- `clahe_unsharp`: CLAHE + unsharp masking para resaltar estructuras.

https://www.kaggle.com/code/zeeshanshaik75/tb-2class-image-enhancement-techniques

Parámetros relacionados:
- `--clahe-clip-limit`
- `--clahe-tile-grid`
- `--gamma`
- `--unsharp-sigma`
- `--unsharp-amount`

## Tecnología de Entrenamiento

El entrenamiento está diseñado para mejorar generalización y confiabilidad clínica:

- **Transfer Learning** con EfficientNet-B4.
- Entrenamiento en 2 fases:
  - fase 1: congelación de backbone + entrenamiento de cabeza clasificadora
  - fase 2: fine-tuning parcial de capas finales
- **Balanceo de clases** con `WeightedRandomSampler`.
- Función de pérdida para clasificación multiclase: `CrossEntropyLoss`.
- Scheduler de learning rate: `ReduceLROnPlateau`.
- Early stopping por métrica de validación.
- Soporte de validación externa (`val_2`) para controlar *domain shift*.
- Selección de umbral configurable:
  - `who_tpp`
  - `strict`
  - `balanced`

## Métricas de Evaluación

El proyecto reporta métricas clínicas y de ML en validación/test:

- AUC
- Sensibilidad (Recall para TB)
- Especificidad
- F1-score
- Balanced Accuracy
- Matriz de confusión
- Reporte de clasificación por clase

Además, guarda resultados en:
- `models/efficientnet_b4_tb_best.pt` (modelo)
- `models/training_metrics.json` (métricas)

## Estructura de datos

- `data1/`
  - `NORMAL/`
  - `TB/`
- `data2/`
  - `NORMAL/`
  - `TB/`

## ¿Qué hace cada archivo en `src/`?

- `src/train.py`
  - Entrena el clasificador TB vs NORMAL con EfficientNet-B4.
  - Aplica enhancement y segmentación pulmonar heurística opcional.
  - Calcula umbral final (`who_tpp`, `strict`, `balanced`).
  - Evalúa en test interno/externo.
  - Guarda modelo y métricas.
  - Puede generar figura Score-CAM comparativa con `--make-cam-grid`.

- `src/generate_cam_grid.py`
  - Genera la grilla de explicabilidad (Score-CAM) **sin reentrenar**.
  - Carga un `.pt` ya entrenado y una imagen CXR nueva.
  - Produce comparación visual de activaciones para distintas técnicas de enhancement.

- `src/split_dataset.py`
  - Prepara el dataset mezclando `data1` y `data2`.
  - Crea `train`, `val_1`, `val_2`, `test_1`, `test_2`.
  - Permite configurar proporciones para validación interna/externa y holdout externo.

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## 1) Generar split (mezcla de dominios + calibracion externa)

Recomendado para confiabilidad (deja mas datos de `data2` para entrenamiento):

```bash
python src/split_dataset.py --target data_prepared_mixed --val-ratio 0.15 --test-ratio 0.45 --external-holdout-ratio 0.30 --external-val-ratio 0.15 --seed 42
```

Esto genera:
- `data_prepared_mixed/train/` (mezcla de `data1` + `data2`, excepto `val_2` y `test_2`)
- `data_prepared_mixed/val_1/` (validacion desde `data1`)
- `data_prepared_mixed/val_2/` (validacion desde `data2`)
- `data_prepared_mixed/test_1/` (test interno desde `data1`)
- `data_prepared_mixed/test_2/` (test externo holdout desde `data2`)

Si necesitas exactamente los conteos antiguos de test:

```bash
python src/split_dataset.py --target data_prepared_mixed --val-ratio 0.15 --test-ratio 0.45 --external-holdout-ratio 0.901 --external-val-ratio 0.15 --seed 42
```

## 2) Entrenamiento y contenido de la salida `.pt`

Entrenamiento recomendado (incluye generación de grilla Score-CAM al final):

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 28 --batch-size 8 --img-size 380 --min-sensitivity 0.90 --min-specificity 0.70 --threshold-policy who_tpp --enhancement-mode clahe_gamma --lung-segmentation-mode heuristic --make-cam-grid
```

Políticas de umbral:
- `who_tpp`: intenta cumplir sensibilidad minima + especificidad minima (recomendado para tamizaje)
- `strict`: prioriza mayor especificidad manteniendo sensibilidad minima
- `balanced`: prioriza balanced accuracy

El archivo `models/efficientnet_b4_tb_best.pt` ya guarda estas claves para reproducibilidad:
- `model_state_dict`: pesos de la red EfficientNet-B4 entrenada.
- `class_to_idx_train`: mapeo de clases usado en entrenamiento.
- `tb_index_train`: índice exacto de la clase `TB`.
- `img_size`: tamaño de entrada usado.
- `threshold`: umbral final calibrado.
- `mean` y `std`: normalización aplicada en inferencia.
- `enhancement_mode`: técnica de enhancement usada al entrenar.
- `lung_segmentation_mode`: modo de segmentación pulmonar (`none` o `heuristic`).
- `lung_segmentation_outside_scale`: intensidad de atenuación fuera del pulmón.

## Salidas

- Modelo: `models/efficientnet_b4_tb_best.pt`
- Metricas: `models/training_metrics.json`
  
## Dataset
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/data
https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip

## Nota

Si Windows bloquea archivos al limpiar carpetas, ejecuta el split sin `--clean` o usa otro `--target`.
