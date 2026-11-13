# 🧠 CIMLAB AI — Código fuente Python de desarrollo

**Snapshot de desarrollo para TF3**, ensamblado a partir del código fuente v0.5 y de las correcciones **v0.5.1–v0.5.4** probadas en este proyecto: diagnóstico de recortes, lectura TIFF RGB `int16`, compatibilidad de muestras y respeto de las particiones `tier1/hold/test`.

> [!IMPORTANT]
> Se proporciona el **código Python**, no un instalador de Windows, un modelo entrenado ni los TIFF. Esta copia se verificó mediante compilación y **tres pruebas sintéticas**. No ha pasado todavía una auditoría final en una computadora independiente con la colección real; los resultados de México + Palu continúan en evaluación.

## Módulos principales

| Archivo | Función |
|:--|:--|
| `app.py` | Interfaz local |
| `event_sources.py` | Fuentes y eventos dinámicos |
| `data_core.py` | Descarga, auditoría, preparación, particiones y recortes |
| `tiff_support.py` | Lectura de TIFF RGB con compatibilidad `int16` |
| `labels_core.py` | Etiquetas binarias y validación de equivalencias |
| `models_core.py` | Entrenamiento y evaluación principal |
| `advanced_core.py` | Modelos avanzados |
| `comparison_core.py` | Comparación de modelos |
| `audit_core.py` | Auditoría visual de predicciones |
| `quality_core.py` | Control de calidad y comparador PRE/POST |
| `predict_core.py` | Predicción por escena |
| `report_core.py` | Reportes y figuras |

## Dependencias

Primero instalar las dependencias de base del archivo `requirements-base.txt` en un **entorno virtual propio**. Para las redes neuronales será necesario instalar PyTorch apropiado para el equipo (CPU o GPU) y las librerías de `requirements-ai.txt`. No se incluye ni se modifica ninguna `.venv` local. Las versiones utilizadas por el grupo deben congelarse y documentarse al cerrar TF3.

```bash
python -m venv .venv
# Activar el entorno según tu sistema operativo
python -m pip install -r requirements-base.txt
# Después: instalar PyTorch apropiado y dependencias opcionales de IA
python app.py
```

**Atención:** el código de escritorio utiliza Tkinter; debe ejecutarse en un equipo con entorno gráfico. El motor incluye el entrenamiento de ProtoNet DINO entre sus arquitecturas. Sus métricas deben guardarse con la configuración de la ejecución antes de publicarlas como definitivas. Los entrenamientos de redes neuronales pueden requerir paquetes de mayor tamaño, y algunos modelos descargan pesos preentrenados desde sus fuentes originales. El código no crea una instalación automática para todas las plataformas.

## Pruebas disponibles

```bash
python tests/label_compat_test.py
python tests/multievent_test.py
python tests/smoke_test.py
```

Estas pruebas utilizan **datos sintéticos** y **no son resultados científicos del TF3**. El test de humo ejecuta una prueba `test` únicamente sobre imágenes ficticias creadas en una carpeta temporal; no toca la prueba reservada real.

## Condiciones de uso

- `0 = no-damage`; `1 = minor-damage`, `major-damage` o `destroyed`.
- `un-classified` queda excluido; etiquetas externas incompatibles deben revisarse.
- `tier1 → train`, `hold → val`, `test → test` para datos xBD.
- No usar el test real para escoger arquitectura, umbral o hiperparámetros.
- Reportar métricas por evento: Palu domina la cantidad de edificios dañados.

El verificador independiente `../auditar_manifest.py` puede revisar el conjunto preparado sin cargar esta aplicación.

### Tratamiento de TIFF y recortes

`tiff_support.py` abre TIFF RGB `int16` con `tifffile` cuando Pillow no los puede leer; convierte exactamente a `uint8` si el rango está en 0–255. `data_core.py` utiliza una ventana común PRE/POST por edificio y relleno por reflexión en bordes que exceden la imagen, evitando franjas negras artificiales. Esta estrategia no crea información geoespacial ni aumenta la resolución.
