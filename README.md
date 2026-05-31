# Detección Automatizada de Tuberculosis en Imágenes de Rayos-X Torácico Usando Fine-Tuning de Redes Neuronales Profundas

Proyecto de clasificación binaria (`TB` vs `NORMAL`) usando transfer learning con EfficientNet-B4 y segmentación pulmonar con U-Net/Attention U-Net.

## Tecnologías Utilizadas

Este proyecto está desarrollado en **Python** y usa un pipeline de **Deep Learning + procesamiento de imágenes médicas** para clasificar radiografías de tórax en dos clases: `NORMAL` y `TB`.

### Stack principal

- **Python 3.10**
- **PyTorch (`torch`)**: framework principal para entrenamiento e inferencia.
- **Torchvision (`torchvision`)**: modelos preentrenados, transformaciones de imagen y `ImageFolder`.
- **EfficientNet-B4**: arquitectura base recomendada para clasificación.
- **DenseNet169**: backbone alternativo disponible.
- **U-Net / Attention U-Net**: segmentación pulmonar previa a la clasificación.
- **Torch-DirectML (`torch-directml`)**: aceleración en GPU AMD.
- **NumPy**: operaciones numéricas.
- **Scikit-learn (`scikit-learn`)**: métricas de evaluación (AUC, F1, matriz de confusión, etc.).
- **OpenCV (`opencv-python`)**: técnicas de mejora de imagen, máscaras y procesamiento CXR.
- **Matplotlib**: generación de grillas Score-CAM.
- **PIL/Pillow**: lectura y manipulación de imágenes.
- **JSON**: almacenamiento de métricas y resultados.

## Tecnología Aplicada a Imágenes (Radiografías CXR)

El proyecto incluye un flujo específico para imágenes médicas de tórax:

- Redimensionamiento de imagen a resolución configurable (`--img-size`, por defecto `320` en `train.py`).
- Conversión de imagen a escala de grises y posterior adaptación a 3 canales (RGB) para modelos preentrenados.
- Normalización con estadísticas de ImageNet.
- Segmentación pulmonar antes de clasificar:
  - `attention_unet` por defecto.
  - `unet` opcional.
  - `heuristic` como respaldo.
  - `none` para desactivar.
- Aumentación de datos controlada:
  - rotación leve (`--rotation-deg`)
  - flip horizontal configurable (`--hflip-prob`)
  - `RandomResizedCrop`
  - ajustes suaves de brillo/contraste
- Filtro de collages con `FilteredImageFolder` salvo que se use `--allow-collage-image`.

### Técnicas de enhancement disponibles

Se pueden activar desde `train.py` con `--enhancement-mode`:

- `none`: sin mejora adicional.
- `histeq`: ecualización de histograma.
- `clahe`: mejora de contraste local con CLAHE.
- `gamma`: corrección gamma.
- `complement`: complemento/inversión de intensidad.
- `bcet`: Balance Contrast Enhancement Technique.
- `clahe_gamma`: CLAHE + corrección gamma (por defecto).
- `clahe_unsharp`: CLAHE + unsharp masking.
- `tricanal`: representación RGB con variantes de mejora.

Referencia de técnicas de enhancement:
https://www.kaggle.com/code/zeeshanshaik75/tb-2class-image-enhancement-techniques

Parámetros relacionados:
- `--clahe-clip-limit`
- `--clahe-tile-grid`
- `--gamma`
- `--bcet-target-mean`
- `--unsharp-sigma`
- `--unsharp-amount`

## Tecnología de Entrenamiento

El entrenamiento está diseñado para mejorar generalización y confiabilidad clínica:

- **Transfer Learning** con backbone preentrenado.
- Backbone por defecto: `efficientnet_b4`.
- Backbones soportados:
  - `efficientnet_b4`
  - `densenet169`
- Entrenamiento en 2 fases:
  - fase 1: congelación del extractor de características + entrenamiento de cabeza clasificadora
  - fase 2: fine-tuning parcial de capas finales
- **Balanceo de clases** con `WeightedRandomSampler`.
- Función de pérdida: `CrossEntropyLoss`.
- Scheduler de learning rate: `ReduceLROnPlateau`.
- Early stopping por métrica de validación.
- Soporte de evaluación externa opcional con `--enable-external-eval`.
- Selección de umbral configurable:
  - `who_tpp`
  - `strict`
  - `balanced`
  - `hybrid` (OMS + mejor F1/BalAcc)
- Selección de checkpoint configurable:
  - `auc`
  - `clinical` (por defecto actual)
- Cache opcional de máscara pulmonar en RAM (`--lung-mask-cache`) para acelerar epochs.
- Score-CAM opcional al final con `--make-cam-grid`.

## Métricas de Evaluación

El proyecto reporta métricas clínicas y de ML en validación/test:

- AUC
- Sensibilidad (Recall para TB)
- Especificidad
- Precisión
- F1-score
- Balanced Accuracy
- Matriz de confusión
- Reporte de clasificación por clase

Además, guarda resultados en:
- `models/<backbone>_tb_best.pt` (modelo clasificador)
- `models/<backbone>_tb_best_training_metrics.json` (métricas)
- `models/<backbone>_tb_best_cam_grid.png` (Score-CAM si se activa)

## Estructura de datos

Dataset de clasificación:

- `data1/`
  - `NORMAL/`
  - `TB/`
- `data2/`
  - `NORMAL/`
  - `TB/`

Dataset de segmentación pulmonar:

- `segmentacion/`
  - `CXR_png/`
  - `masks/` (máscara bilateral en una sola imagen)

Dataset preparado por `split_dataset.py`:

- `data_prepared_mixed/`
  - `train/`
    - `NORMAL/`
    - `TB/`
  - `val_1/`
    - `NORMAL/`
    - `TB/`
  - `test_1/`
    - `NORMAL/`
    - `TB/`

## ¿Qué hace cada archivo en `src/`?

- `src/split_dataset.py`
  - Prepara el dataset mezclando `data1` y `data2`.
  - Balancea `NORMAL` y `TB` usando el mínimo disponible o `--target-per-class`.
  - Crea `train`, `val_1`, `test_1`.
  - Estratifica por dataset de origen para mantener proporciones estables.
  - Permite `--balance-domain`, `--deduplicate` y `--quality-filter`.
  - Copia archivos con nombres estables que incluyen dominio (`d1`, `d2`).

- `src/train_lung_unet.py`
  - Entrena el segmentador pulmonar.
  - Usa `segmentacion/CXR_png` como imágenes.
  - Usa `segmentacion/mask` o `segmentacion/masks` como máscara pulmonar bilateral.
  - Entrena `attention_unet` por defecto o `unet`.
  - Usa pérdida combinada Dice + BCE.
  - Reporta Dice e IoU.
  - Guarda `models/lung_attention_unet_best.pt`.

- `src/train.py`
  - Entrena el clasificador `TB` vs `NORMAL`.
  - Aplica enhancement.
  - Aplica segmentación pulmonar con U-Net/Attention U-Net antes de clasificar.
  - Si falta checkpoint U-Net, usa segmentación heurística como respaldo.
  - Soporta `efficientnet_b4` y `densenet169`.
  - Calcula umbral final (`who_tpp`, `strict`, `balanced`, `hybrid`).
  - Selecciona mejor checkpoint por `auc` o por score clínico (`--selection-policy clinical`).
  - Permite cache de máscara pulmonar (`--lung-mask-cache`) para acelerar entrenamiento.
  - Evalúa en test interno y externo opcional.
  - Guarda modelo y métricas.
  - Puede generar figura Score-CAM con `--make-cam-grid`.

- `src/generate_cam_grid.py`
  - Genera grilla Score-CAM **sin reentrenar**.
  - Carga un `.pt` ya entrenado y una imagen CXR nueva.
  - Reutiliza `generate_cam_grid_figure` desde `train.py`.
  - Aplica segmentación pulmonar configurada.
  - Produce comparación visual de activaciones para distintas técnicas de enhancement.

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## 1) Generar split balanceado

Recomendado para confiabilidad:

```bash
python src/split_dataset.py --target data_prepared_mixed --clean --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Esto genera:
- `data_prepared_mixed/train/`
- `data_prepared_mixed/val_1/`
- `data_prepared_mixed/test_1/`

Con los datos actuales, el script usa automáticamente el mínimo disponible entre clases. Ejemplo:

- `train`: `NORMAL=276`, `TB=276`
- `val_1`: `NORMAL=59`, `TB=59`
- `test_1`: `NORMAL=59`, `TB=59`

Opciones útiles:

```bash
python src/split_dataset.py --target data_prepared_mixed --clean --target-per-class 300 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

```bash
python src/split_dataset.py --target data_prepared_mixed --clean --balance-domain --seed 42
```

`--balance-domain` fuerza igualdad entre `data1` y `data2`, pero puede usar menos imágenes porque `data1` tiene menos muestras.

## 2) Entrenar segmentador pulmonar U-Net

Primero entrena el Attention U-Net con las máscaras manuales:

```bash
python src/train_lung_unet.py --segmentation-dir segmentacion --model-dir models --output-name lung_attention_unet_best.pt --architecture attention_unet --img-size 320 --epochs 40 --batch-size 4
```

Salida:
- `models/lung_attention_unet_best.pt`
- `models/lung_attention_unet_best_metrics.json`

En DirectML por memoria GPU. Usa `320` o `512`.

Referencia de segmentación pulmonar:

https://www.kaggle.com/code/iamtapendu/attention-u-net-lungs-segmentation-classification

## 3) Entrenamiento y contenido de la salida `.pt`

Entrenamiento recomendado (incluye segmentación pulmonar + selección clínica + Score-CAM):

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 60 --batch-size 8 --num-workers 0 --img-size 380 --backbone efficientnet_b4 --output-name efficientnet_b4_tb_best.pt --enhancement-mode clahe_gamma --lung-segmentation-mode attention_unet --lung-unet-checkpoint models/lung_attention_unet_best.pt --lung-mask-cache --lung-mask-cache-max-items 2500 --min-sensitivity 0.90 --min-specificity 0.70 --threshold-policy hybrid --selection-policy clinical --make-cam-grid
```

`src/train.py` permite elegir arquitectura con `--backbone`:

- `efficientnet_b4`: opción recomendada por defecto.
- `densenet169`: alternativa preentrenada para comparar rendimiento.

Comando usando EfficientNet-B4:

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 60 --batch-size 8 --num-workers 0 --img-size 380 --backbone efficientnet_b4 --output-name efficientnet_b4_tb_best.pt --enhancement-mode clahe_gamma --lung-segmentation-mode attention_unet --lung-unet-checkpoint models/lung_attention_unet_best.pt --lung-mask-cache --threshold-policy hybrid --selection-policy clinical --make-cam-grid
```

Comando usando DenseNet169:

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 60 --batch-size 8 --num-workers 0 --img-size 380 --backbone densenet169 --output-name densenet169_tb_best.pt --enhancement-mode clahe_gamma --lung-segmentation-mode attention_unet --lung-unet-checkpoint models/lung_attention_unet_best.pt --lung-mask-cache --threshold-policy hybrid --selection-policy clinical --make-cam-grid
```

Si quieres subir resolución:

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 28 --batch-size 4 --img-size 512 --backbone efficientnet_b4 --enhancement-mode clahe_gamma --lung-segmentation-mode attention_unet --lung-unet-checkpoint models/lung_attention_unet_best.pt --make-cam-grid
```

Políticas de umbral:
- `who_tpp`: intenta cumplir sensibilidad mínima + especificidad mínima.
- `strict`: prioriza mayor especificidad manteniendo sensibilidad mínima.
- `balanced`: prioriza balanced accuracy.
- `hybrid`: exige piso OMS y luego prioriza F1/Balanced Accuracy.

Políticas de selección de checkpoint:
- `auc`: guarda mejor por AUC.
- `clinical`: guarda mejor por score clínico (AUC+F1+BalAcc con penalización por incumplir OMS).

El archivo `models/<backbone>_tb_best.pt` guarda claves para reproducibilidad:
- `model_state_dict`: pesos del clasificador entrenado.
- `class_to_idx_train`: mapeo de clases usado.
- `tb_index_train`: índice exacto de clase `TB`.
- `img_size`: tamaño de entrada usado.
- `threshold`: umbral final calibrado.
- `mean` y `std`: normalización aplicada.
- `enhancement_mode`: técnica de enhancement.
- `lung_segmentation_mode`: modo de segmentación pulmonar (`none`, `heuristic`, `unet`, `attention_unet`).
- `lung_segmentation_outside_scale`: atenuación fuera del pulmón.
- `lung_unet_checkpoint`: checkpoint del segmentador.
- `backbone`: arquitectura usada.
- `apical_weight`: ponderación apical para Score-CAM.

## 4) Generar Score-CAM sin reentrenar

```bash
python src/generate_cam_grid.py --model-path models/efficientnet_b4_tb_best.pt --image-path ruta/a/radiografia.png --output-path models/tb_enhancement_cam_grid.png --lung-segmentation-mode attention_unet --lung-unet-checkpoint models/lung_attention_unet_best.pt
```

## Salidas

- Segmentador: `models/lung_attention_unet_best.pt`
- Métricas segmentador: `models/lung_attention_unet_best_metrics.json`
- Clasificador: `models/<backbone>_tb_best.pt`
- Métricas clasificador: `models/<backbone>_tb_best_training_metrics.json`
- Explicabilidad: `models/<backbone>_tb_best_cam_grid.png`

## Dataset 

https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities/data

## Nota

Si Windows bloquea archivos al limpiar carpetas, ejecuta el split sin `--clean` o usa otro `--target`.

Si aparece error de memoria en DirectML, baja `--img-size` o `--batch-size`.
