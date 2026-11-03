<div align="center">

# INFORME DE TRABAJO N.° 2
## Análisis exploratorio de datos y plan algorítmico
### Evaluación rápida de daño estructural post-sismo mediante imágenes satelitales

**Curso:** Aplicaciones de IA en Estructuras  
**Docente:** Ing. Kurt Walter Soncco Sinchi  
**Grupo:** 04

**Integrantes:** Jhon Héctor Aire Artezano · Kevin Flores Paniura · Katherine Limaymanta Caballero · Jairo Paredes Mercado

</div>

---

## I. Análisis exploratorio de datos (EDA)

### 1.1. Conjunto de datos

Se utiliza el dataset **xBD / xView2**, que reúne imágenes satelitales anteriores y posteriores a desastres, junto con polígonos de edificaciones y etiquetas de daño [1]. El caso de estudio corresponde al evento `mexico-earthquake`, asociado con el sismo de México de 2017.

El procesamiento inicial reunió anotaciones de **193 escenas**. Se revisó la correspondencia de identificadores de edificación en **193 archivos JSON PRE y 193 POST**, obteniendo una tabla de **51 473 registros**.

| Variable | Descripción |
|:--|:--|
| `escena` | Código de la escena satelital |
| `uid` | Identificador de la edificación |
| `clasificacion_dano` | Nivel de daño asignado a la edificación |

**Archivos:** [Tabla de edificaciones](01_Datos_Preparados/resultados/tabla_edificaciones.csv) · [Validación de escenas](01_Datos_Preparados/resultados/validacion_escenas.csv) · [Código de preparación](01_Datos_Preparados/scripts/validacion_dataset.py).

### 1.2. Validación y limpieza

La revisión se realizó en Python, mediante controles de valores vacíos, registros repetidos, formato de identificadores y categorías de daño. Los resultados fueron los siguientes:

| Indicador | Cantidad |
|:--|--:|
| Registros analizados | **51 473** |
| Escenas identificadas | **193** |
| Registros con campos vacíos | **0** |
| Filas duplicadas | **0** |
| Identificadores repetidos dentro de una escena | **0** |
| Identificadores con formato inválido | **0** |
| Etiquetas no reconocidas | **0** |
| Registros `un-classified` | **111** |

No se detectaron inconsistencias en los campos evaluados. Para la clasificación supervisada se separaron los registros `un-classified`, cuya categoría de daño no está definida. Se conservó la tabla original.

| Resultado de la separación | Registros |
|:--|--:|
| Edificaciones con clase de daño conocida | **51 362** |
| Edificaciones sin clasificar | **111** |
| Registros observados por errores de formato o duplicación | **0** |

Este control corresponde a los campos del CSV; la inspección de imágenes y polígonos se realizará con los archivos originales.

**Código y resultados:** [Validación y limpieza](01_Datos_Preparados/scripts/validar_limpiar_edificaciones.py) · [Reporte de auditoría](01_Datos_Preparados/resultados/resumen_auditoria.txt).

### 1.3. Distribución de daños

| Clasificación | Edificaciones | Porcentaje |
|:--|--:|--:|
| `no-damage` — Sin daño | **51 084** | 99.2443 % |
| `minor-damage` — Daño menor | **221** | 0.4294 % |
| `major-damage` — Daño mayor | **54** | 0.1049 % |
| `destroyed` — Destruida | **3** | 0.0058 % |
| `un-classified` — Sin clasificar | **111** | 0.2156 % |
| **Total** | **51 473** | **100 %** |

Predomina la clase `no-damage`, que representa el **99.24 %** de los registros. La representación de las categorías de daño es considerablemente menor, en particular la clase `destroyed`, con **tres casos**.

#### Distribución completa

<div align="center">

![Distribución completa de daños](01_Datos_Preparados/resultados/EDA/01_distribucion_danos.png)

*Figura 1. Distribución de las categorías de daño en escala logarítmica.*

</div>

#### Detalle de categorías minoritarias

<div align="center">

![Categorías minoritarias](01_Datos_Preparados/resultados/EDA/02_detalle_categorias_minoritarias.png)

*Figura 2. Categorías minoritarias; se omite «sin daño» únicamente de esta representación gráfica.*

</div>

### 1.4. Distribución por escena

<div align="center">

![Edificaciones por escena](01_Datos_Preparados/resultados/EDA/03_edificaciones_por_escena.png)

*Figura 3. Número de edificaciones anotadas por escena.*

</div>

| Indicador | Resultado |
|:--|--:|
| Escenas analizadas | **193** |
| Promedio de edificaciones por escena | **266.7** |
| Mediana | **237** |
| Mínimo | **2** |
| Máximo | **1 157** |
| Escenas con al menos una edificación dañada | **47** |

El número de edificaciones varía considerablemente entre escenas. Esta diferencia se tendrá en cuenta al organizar los conjuntos de entrenamiento, validación y prueba.

**Código y resultados:** [Análisis exploratorio](01_Datos_Preparados/scripts/analisis_exploratorio.py) · [Distribución de daños](01_Datos_Preparados/resultados/EDA/distribucion_eda.csv) · [Conteos por escena](01_Datos_Preparados/resultados/EDA/edificaciones_por_escena.csv) · [Resumen](01_Datos_Preparados/resultados/EDA/resumen_eda.txt).

### 1.5. Observaciones del EDA

Los resultados permiten identificar tres aspectos relevantes para el modelado:

1. **Desbalance de clases:** las edificaciones sin daño concentran casi todos los registros. Será necesario evaluar estrategias de entrenamiento que consideren las clases minoritarias.
2. **Escasez de edificios destruidos:** los tres casos disponibles resultan insuficientes para evaluar de forma robusta esa categoría. Se revisará la incorporación de datos adicionales.
3. **Variables disponibles:** el CSV actual incluye identificadores y etiquetas, pero no áreas ni otras variables geométricas. El análisis de esas características requiere revisar los polígonos de los JSON originales.

La siguiente revisión comprenderá pares de imágenes PRE/POST, integridad de archivos y geometrías de las edificaciones. A partir de estas últimas se podrán obtener estadísticas de área, según el sistema de coordenadas disponible.

---

## II. Plan algorítmico

### 2.1. Objetivo del modelo

Se propone clasificar el **daño visible de una edificación** a partir de un par de imágenes satelitales del mismo sector: una anterior y otra posterior al sismo.

| Componente | Definición |
|:--|:--|
| Entrada | Recortes RGB PRE y POST de una misma edificación |
| Salida | `no-damage`, `minor-damage`, `major-damage` o `destroyed` |
| Enfoque | Aprendizaje supervisado mediante redes neuronales convolucionales |
| Herramientas | Python y PyTorch |

```text
Imagen PRE ──┐
             ├──► Recortes de la edificación ──► Modelo CNN ──► Nivel de daño
Imagen POST ─┘
```

### 2.2. Modelos propuestos

| Modelo | Descripción | Finalidad de la comparación |
|:--|:--|:--|
| **ResNet18 con fusión temprana** | Combina las imágenes PRE y POST en una entrada de seis canales. | Evaluar una arquitectura de clasificación conjunta. |
| **ResNet18 siamesa** | Procesa las dos imágenes en ramas separadas y fusiona las características extraídas. | Evaluar el aporte de comparar representaciones visuales de ambos momentos. |

Ambos modelos se plantean como alternativas de complejidad moderada para el trabajo del grupo. Las redes de dos entradas se relacionan con enfoques utilizados en el desafío xView2 [3, 4].

### 2.3. Preparación de las imágenes

El procedimiento previsto comprende:

1. Verificar la integridad de las imágenes y su correspondencia PRE/POST.
2. Extraer recortes por edificación utilizando las anotaciones de los polígonos.
3. Ajustar tamaño y normalización de las imágenes de entrada.
4. Dividir los datos **por escenas** para evitar compartir edificaciones de una misma escena entre entrenamiento y prueba.
5. Evaluar ponderación de clases, muestreo o *focal loss* para atender el desbalance durante el entrenamiento.

Antes de entrenar un clasificador de cuatro categorías se revisará la disponibilidad de ejemplos de daño severo, ya que el conjunto actual contiene solamente tres edificios destruidos.

---

## III. Métricas de evaluación

Dada la distribución observada, se priorizarán indicadores que permitan estudiar el desempeño de cada categoría de daño.

| Métrica | Uso previsto |
|:--|:--|
| **F1-score macro** | Comparar el desempeño global dando el mismo peso a cada clase. |
| **Recall por clase** | Medir cuántas edificaciones de cada categoría son identificadas correctamente. |
| **Precision por clase** | Evaluar la proporción de aciertos dentro de las predicciones de cada categoría. |
| **Matriz de confusión** | Identificar confusiones entre niveles de daño. |
| **Accuracy** | Complementar la evaluación con el porcentaje general de aciertos. |

La exactitud global, por sí sola, sería poco representativa: en los **51 362 registros clasificados**, una predicción constante de «sin daño» alcanzaría aproximadamente **99.46 % de accuracy** sin identificar ninguno de los casos dañados.

---

## IV. Marco PCS

Se consideran los principios de **Predictibilidad, Computabilidad y Estabilidad** del enfoque *Veridical Data Science* [2].

| Principio | Aplicación al proyecto |
|:--|:--|
| **Predictibilidad** | Comparación de modelos mediante F1 macro, recall por clase y matriz de confusión. |
| **Computabilidad** | Ejecución reproducible de la preparación, validación y análisis en Python. |
| **Estabilidad** | Revisión de resultados entre particiones por escena y distintas condiciones de entrenamiento. |

---

## V. Archivos del proyecto

| Archivo | Contenido |
|:--|:--|
| [`validacion_dataset.py`](01_Datos_Preparados/scripts/validacion_dataset.py) | Preparación y correspondencia de anotaciones PRE/POST. |
| [`validar_limpiar_edificaciones.py`](01_Datos_Preparados/scripts/validar_limpiar_edificaciones.py) | Auditoría y separación de registros sin clasificar. |
| [`analisis_exploratorio.py`](01_Datos_Preparados/scripts/analisis_exploratorio.py) | Estadísticas y gráficos del EDA. |
| [`resultados/EDA`](01_Datos_Preparados/resultados/EDA) | Gráficos, tablas y resumen estadístico. |

Para ejecutar el EDA desde la carpeta principal del repositorio:

```bash
pip install pandas matplotlib
python 01_Datos_Preparados/scripts/analisis_exploratorio.py
```

---

## VI. Referencias

1. **Gupta, R., et al. (2019).** *Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery*. CVPR Workshops. [Artículo](https://arxiv.org/abs/1911.09296) · [xView2](https://www.xview2.org/dataset).
2. **Yu, B., & Barter, R. (2024).** *Veridical Data Science*. MIT Press. [Libro](https://vdsbook.com/).
3. **DIU xView2.** *xView2 Baseline: Building Localization and Damage Classification*. [Repositorio](https://github.com/DIUx-xView/xView2_baseline).
4. **DIU xView2.** *First Place Solution for xView2*. [Repositorio](https://github.com/DIUx-xView/xView2_first_place).
