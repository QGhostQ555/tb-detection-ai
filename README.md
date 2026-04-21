# Clasificacion de Tuberculosis Pulmonar en Radiografias de Torax (EfficientNet-B4)

Proyecto de clasificacion binaria (`TB` vs `NORMAL`) usando transfer learning con EfficientNet-B4.

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
