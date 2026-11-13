# 🔧 Guía de reproducibilidad de TF3

**Estado:** preparación conjunta validada; entrenamiento y evaluación multievento en desarrollo. No equivale todavía a un paquete de reproducción de modelos completo.

## 1. Repositorio y descargas

Clonar el repositorio del Grupo 04, con Git LFS instalado, y obtener los datos reales de `data/mexico-earthquake/` y `data/palu-tsunami/`. Cada evento contiene pares `*_pre_disaster.tif`, `*_post_disaster.tif` y JSON con los polígonos de edificios.

Los archivos TIFF se gestionan mediante Git LFS. Un archivo puntero LFS no es una imagen aunque termine en `.tif`. Verificar el contenido y la decodificación antes de preparar recortes. No es necesario descargar archivos de otros eventos para ejecutar exclusivamente México y Palu.

## 2. Procesamiento y clases

La tarea se define como clasificación binaria de **cualquier daño visible**. `no-damage → 0`, `minor-damage → 1`, `major-damage → 1`, `destroyed → 1`. `un-classified` se excluye. No interpretar etiquetas desconocidas automáticamente.

Los TIFF RGB se encontraron como `int16` en el entorno utilizado. El lector v0.5.3 verifica el rango de cada archivo y permite una conversión exacta a RGB `uint8` cuando todos sus valores están en [0, 255], sin alterar el brillo o los colores. El recorte utiliza la misma ventana para PRE y POST (12 píxeles de contexto, salida 224 × 224); si la ventana cruza el límite de imagen, completa la zona exterior por reflexión de píxeles del borde, no con negro. Esto evita franjas artificiales, **pero no recupera detalles del terreno ausentes de la captura ni registra geométricamente las imágenes**. Los recortes conservan las referencias a polígonos y metadatos originales en el manifiesto.

## 3. Separación entrenamiento / validación / prueba

**No aleatorizar nuevamente todas las escenas.** Usar `tier1 → train`, `hold → val` y `test → test` del repositorio original. Los recortes locales fueron preparados antes de detectar una asignación incorrecta de particiones; CIMLAB AI v0.5.4 corrigió el archivo `manifest.csv` sin regenerar los recortes. El script `codigo/auditar_manifest.py` puede verificar las particiones finales.

| Partición | Edificios | Dañados | Escenas |
|---|---:|---:|---:|
| train | 63,189 | 5,668 | 231 |
| val | 19,537 | 1,389 | 74 |
| test | 23,814 | 1,603 | 80 |

## 4. Auditoría Python independiente

Desde la raíz del repositorio:

```bash
python TF3/codigo/auditar_manifest.py "J:/TF 3/Trabajo_Mexico_y_Palu/manifest.csv" --salida "resumen_manifest_tf3.json"
```

Requiere únicamente bibliotecas estándar de Python y no mueve los JPG. Comprueba `scene`, `uid`, `label`, `split`, etiqueta original, evento y partición de rutas PRE/POST. Para validar la existencia física de todos los recortes hace falta ejecutarlo en el equipo de origen con `--verificar-recortes`; el paquete de GitHub no incluye JPG.

Si utilizas el `manifest.csv` **antiguo**, anterior a v0.5.4, obtendrás errores de partición; ejecuta la auditoría sobre el manifiesto **corregido** en la computadora de trabajo.

## 5. Entrenamientos

Para cada corrida registrar: commit del código, arquitectura, versión del manifiesto, semilla, librerías, dispositivo (CPU/GPU), fecha, hiperparámetros, tiempo, tamaño de train y métricas de validación. Conservar el modelo y predicciones por edificio para una comparación reproducible.

Los modelos se seleccionan usando `val`; reservar el conjunto `test` hasta cerrar decisiones. Comparar globalmente y por cada evento (México/Palu). **No ajustar umbrales con `test`.**

## 6. Publicación y límites

Evitar subir carpetas `.venv`, claves, enormes carpetas de recortes, archivos de pesos, manifiestos con rutas absolutas o archivos temporales. El snapshot Python del motor de modelos está incluido en `codigo/cimlab_ai/`, con sus dependencias de partida. Queda **pendiente** fijar todas las versiones en otro equipo, reproducir los entrenamientos reales y publicar los resultados definitivos. No afirmar que el artículo o la reproducción integral están completos hasta verificarlo en otra computadora.
