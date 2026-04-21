# Clasificacion de Tuberculosis Pulmonar en Radiografias de Torax (EfficientNet-B4)

Proyecto de clasificacion binaria (`TB` vs `NORMAL`) usando transfer learning con EfficientNet-B4.

## Tecnologías Utilizadas

Este proyecto está desarrollado en **Python** y usa un pipeline de **Deep Learning + procesamiento de imágenes médicas** para clasificar radiografías de tórax en dos clases: `NORMAL` y `TB`.

### Stack principal

- **Python 3.x**
- **PyTorch (`torch`)**: framework principal para entrenamiento e inferencia.
- **Torchvision (`torchvision`)**: modelos preentrenados y transformaciones de imagen.
- **EfficientNet-B4**: arquitectura base de clasificación (Transfer Learning).
- **Torch-DirectML (`torch-directml`)**: aceleración en GPU AMD (ej. RX5600XT en Windows).
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

Parámetros relacionados:
- `--clahe-clip-limit`
- `--clahe-tile-grid`
- `--gamma`
- `--unsharp-sigma`
- `--unsharp-amount`

## Tecnología de Entrenamiento (Confiabilidad y Robustez)

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
- `data_prepared_mixed/val/` (validacion desde `data1`)
- `data_prepared_mixed/val_2/` (validacion desde `data2`)
- `data_prepared_mixed/test_1/` (test interno desde `data1`)
- `data_prepared_mixed/test_2/` (test externo holdout desde `data2`)

Si necesitas exactamente los conteos antiguos de test:

```bash
python src/split_dataset.py --target data_prepared_mixed --val-ratio 0.15 --test-ratio 0.45 --external-holdout-ratio 0.901 --external-val-ratio 0.15 --seed 42
```

## 2) Entrenar una corrida (confiable y estricta)

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 28 --batch-size 8 --img-size 380 --min-sensitivity 0.90 --min-specificity 0.70 --threshold-policy who_tpp --val2-weight 0.5
```

Para aplicar tecnicas de enhancement (similar a notebooks de Kaggle de TB):

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 28 --min-sensitivity 0.90 --min-specificity 0.70 --threshold-policy strict --enhancement-mode clahe_unsharp --clahe-clip-limit 2.0 --clahe-tile-grid 8 --unsharp-sigma 1.0 --unsharp-amount 1.0
```

Para RX5600XT (DirectML):

```bash
python src/train.py --data-dir data_prepared_mixed --epochs 28 --min-sensitivity 0.90 --min-specificity 0.70 --threshold-policy strict --val2-weight 0.5 --fast-amd --num-workers 0
```

Politicas de umbral:
- `who_tpp`: intenta cumplir sensibilidad minima + especificidad minima (recomendado para tamizaje)
- `strict`: prioriza mayor especificidad manteniendo sensibilidad minima
- `balanced`: prioriza balanced accuracy

Enhancement disponibles:
- `none`
- `clahe`
- `clahe_gamma`
- `clahe_unsharp`

## Salidas

- Modelo: `models/efficientnet_b4_tb_best.pt`
- Metricas: `models/training_metrics.json`

## Nota

Si Windows bloquea archivos al limpiar carpetas, ejecuta el split sin `--clean` o usa otro `--target`.
