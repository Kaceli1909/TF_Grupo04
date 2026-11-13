# 📊 Registro de resultados — Validación México + Palu

**Corte:** 12 de noviembre de 2026  
**Tarea:** dañado (1) frente a no dañado (0)  
**Partición evaluada:** `hold` · 19,537 edificios (1,389 con daño)

| Modelo | AP | F1 | Recall | Precisión | Condición |
|:--|--:|--:|--:|--:|:--|
| Random Forest | 0.5999 | 0.5703 | 0.5169 | 0.6360 | Entrenamiento finalizado |
| DINOv2 + XGBoost | 0.7400 | 0.6897 | 0.6033 | 0.8050 | Entrenamiento finalizado |
| ProtoNet DINO | 0.7195 | 0.6784 | 0.6386 | 0.7235 | Entrenamiento finalizado |
| Hybrid | 0.7768 | 0.7341 | 0.6796 | 0.7980 | Ensamble automático actualizado |

**Hybrid:** promedio simple de probabilidades de daño provenientes de modelos con archivos de validación compatibles. La composición específica depende de los modelos disponibles al generarlo y se registra en `hybrid_config.json` de la instalación que produjo el resultado. Este archivo no se incluye en la documentación proporcionada.

## ProtoNet DINO y ensamble actualizado

Se registraron métricas de validación de ProtoNet DINO: **AP 0.7195, F1 0.6784, recall 0.6386 y precisión 0.7235**. La última comparación de CIMLAB AI registró Hybrid con **AP 0.7768, F1 0.7341, recall 0.6796 y precisión 0.7980**. La versión anterior de Hybrid había registrado AP 0.7667 y F1 0.7293.

La composición de Hybrid depende de los modelos disponibles cuando se genera. Sin el archivo `hybrid_config.json` no se especifican sus integrantes exactos ni se atribuye la variación a un modelo particular.

## Matrices de confusión — validación conjunta

| Modelo | TP | FP | FN | TN |
|:--|--:|--:|--:|--:|
| Random Forest | 718 | 411 | 671 | 17737 |
| DINOv2 + XGBoost | 838 | 203 | 551 | 17945 |
| ProtoNet DINO | 887 | 339 | 502 | 17809 |
| Hybrid | 944 | 239 | 445 | 17909 |

Los conteos de ProtoNet DINO se tomaron del registro original. Los otros tres son valores enteros reconstruidos de las métricas con cuatro decimales sobre 1,389 positivos y 18,148 negativos. Se comprobó que reproducen la precisión, el recall y el F1 mostrado en CIMLAB AI. Las matrices se encuentran en `resultados/figuras/` y pueden regenerarse con `codigo/graficar_matrices_confusion.py`.

## ResNet18 Early Fusion — entrenamiento no finalizado

Parámetros indicados para la ejecución: **20 épocas máximas, batch size 8 y CUDA activado**. En el registro se observaron:

| Época | Pérdida | AP validación | F1 validación |
|--:|--:|--:|--:|
| 1/20 | 0.2623 | 0.8135 | 0.7709 |
| 2/20 | 0.1927 | 0.7432 | 0.6662 |

Son métricas intermedias del entrenamiento; no se consideran un modelo finalizado ni se comparan en igualdad de condiciones con los cuatro resultados de arriba.

## Condiciones y límites del registro

Los valores se transcribieron de la interfaz de CIMLAB AI. No se adjuntaron curvas completas, matrices de confusión, umbrales, archivos de predicciones por edificio ni pesos de los modelos. **No se ha utilizado `test`**. El entrenamiento combinado de los dos eventos no prueba por sí mismo la generalización a cada desastre; Palu reúne el 96.8 % de los casos dañados del dataset.
