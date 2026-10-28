<div align="center">

# 🛰️ Evaluación rápida de daño estructural post-sismo

**Clasificación preliminar de daños visibles en edificaciones mediante imágenes satelitales e inteligencia artificial**

![Curso](https://img.shields.io/badge/Curso-IA%20en%20Estructuras-17365D?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-xBD%20%7C%20xView2-008C95?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-Planificado-3776AB?style=for-the-badge)
![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-D89B16?style=for-the-badge)

**Grupo 04** · Aplicaciones de IA en Estructuras  
**Docente:** Ing. Kurt Walter Soncco Sinchi

</div>

---

> [!IMPORTANT]
> **Propósito:** investigar técnicas de visión artificial para identificar edificaciones y clasificar su **daño visible** comparando imágenes anteriores y posteriores a desastres. El resultado sería un apoyo para **priorizar inspecciones**, no una evaluación concluyente de seguridad estructural.

## 📑 Contenido

1. [🎯 Problema de ingeniería estructural](#problema)
2. [🛰️ Dataset xBD / xView2](#dataset)
3. [🔄 Metodología propuesta](#metodologia)
4. [🧠 Marco VDS — Principios PCS](#vds)
5. [📅 Entregas del curso](#entregas)
6. [🤝 Colaboración y uso de IA](#colaboracion)
7. [📚 Referencias](#referencias)

<a id="problema"></a>

## 🎯 1. Problema de ingeniería estructural

Después de un sismo severo, las entidades de respuesta necesitan saber **qué edificaciones están dañadas y con qué severidad** para priorizar rescate, evacuación, inspecciones detalladas y habilitación de albergues. Hoy esa evaluación se realiza principalmente mediante inspecciones de campo, que pueden ser lentas y peligrosas y demandar muchos especialistas durante las primeras 24–48 horas.

Lima Metropolitana se ubica en una zona de alta sismicidad y concentra numerosas viviendas autoconstruidas, por lo que un sismo mayor podría generar miles de edificaciones por evaluar al mismo tiempo.

### ❓ Pregunta de investigación

> ¿Es posible clasificar de forma confiable el **nivel de daño visible** de las edificaciones (sin daño, daño menor, daño mayor o destruida) a partir de pares de imágenes satelitales **pre y postevento**, de manera que sirva como **triaje inicial** antes de la inspección de campo?

**Contexto de aplicación:** Lima Metropolitana.  
**Caso de estudio propuesto:** terremoto de Ciudad de México del **19 de septiembre de 2017**. El análisis del evento mexicano no demuestra por sí mismo la validez de las predicciones para Lima.

<a id="dataset"></a>

## 🛰️ 2. Dataset xBD / xView2

La base de datos **xBD**, vinculada al desafío **xView2**, contiene imágenes satelitales anteriores y posteriores a distintos desastres, con anotaciones de edificios y etiquetas de daño.

| Característica | Descripción |
|:--|:--|
| **Fuente** | [xBD / xView2](https://www.xview2.org/dataset) |
| **Referencia** | [Gupta et al. (2019)](https://arxiv.org/abs/1911.09296) |
| **Entradas** | Imágenes satelitales RGB pre y postdesastre |
| **Resolución** | Alta resolución (menor a 0.8 m/píxel, según descripción utilizada por el grupo) |
| **Anotaciones** | Polígonos de edificaciones y categorías de daño |
| **Cobertura de xBD completo** | 19 eventos de desastre |
| **Tamaño aproximado de xBD completo** | 22 000 imágenes y 850 000 polígonos de edificios |
| **Caso sísmico elegido** | Ciudad de México, 19/09/2017 |
| **Cantidad de datos que utilizará el grupo** | **Por definir**, tras revisar los archivos disponibles |

> [!NOTE]
> Las cifras anteriores corresponden al **dataset completo**, no solamente a las imágenes del terremoto de México. La muestra seleccionada y sus cantidades deberán confirmarse durante la exploración de datos.

### 🏚️ Categorías de daño

| Etiqueta | Categoría | Interpretación preliminar |
|:--:|:--|:--|
| 🟢 | **Sin daño** | Sin evidencia importante de daño visible |
| 🟡 | **Daño menor** | Afectación visible relativamente leve |
| 🟠 | **Daño mayor** | Evidencia de afectación significativa |
| 🔴 | **Destruida** | Destrucción extensa de la edificación |

Estas etiquetas **no equivalen** a medir directamente la resistencia ni la estabilidad de una estructura.

### 📋 Tipos de datos y variables

| Dato | Tipo | Uso propuesto |
|:--|:--|:--|
| Imagen previa | RGB | Referencia del estado anterior |
| Imagen posterior | RGB | Identificación de cambios visibles |
| Polígono del edificio | Geometría/anotación | Localización de la edificación |
| Categoría de daño | Categórica | Variable objetivo |
| Evento de desastre | Metadato | Filtrado y comparación de resultados |

<a id="metodologia"></a>

## 🔄 3. Metodología propuesta

<div align="center">

**🛰️ Imágenes pre/post → 🏘️ Edificios anotados → 🧹 Preparación → 🤖 Modelos → 📊 Evaluación → 🚨 Priorización**

</div>

1. **Selección:** elegir imágenes y anotaciones correspondientes al evento de interés.
2. **Exploración (EDA):** estudiar cantidad, distribución y calidad de etiquetas e imágenes.
3. **Preparación:** construir particiones de entrenamiento y prueba sin imágenes duplicadas o relacionadas que distorsionen la evaluación.
4. **Modelado:** seleccionar y entrenar algoritmos de visión artificial.
5. **Evaluación:** comparar las predicciones con las etiquetas y discutir los errores y límites de aplicación.

> **Estado actual:** metodología **prevista**. Todavía no se reportan resultados experimentales ni modelos entrenados.

<a id="vds"></a>

## 🧠 4. Marco VDS — Principios PCS

El trabajo considera los principios del marco **Veridical Data Science (VDS)**, mediante el enfoque **PCS: Predictibilidad, Computabilidad y Estabilidad**, solicitado en el curso.

### 🎯 4.1. Predictibilidad

El dataset xBD contiene imágenes satelitales de edificaciones antes y después de diferentes desastres naturales, acompañadas de etiquetas que describen sus niveles de daño visible.

La disponibilidad de imágenes y clasificaciones previamente establecidas permite plantear un problema de **aprendizaje supervisado**, en el que los algoritmos identifiquen patrones asociados con las diferentes categorías de daño.

Se espera analizar la capacidad predictiva de los modelos mediante métricas de clasificación, prestando especial atención a la identificación de edificaciones con **daños mayores y destruidas**.

### 💻 4.2. Computabilidad

El conjunto de datos xBD dispone de imágenes satelitales y archivos de anotaciones que pueden procesarse mediante herramientas de **Python**.

Para el desarrollo del proyecto se prevé utilizar bibliotecas como **NumPy, Pandas, Matplotlib, PyTorch** y herramientas de procesamiento de imágenes.

Debido al tamaño del dataset y a los recursos computacionales que requiere el entrenamiento de redes neuronales, inicialmente se propone trabajar con una **muestra representativa** de imágenes correspondientes al terremoto de México de 2017.

La cantidad definitiva de imágenes se establecerá después de verificar su disponibilidad y la distribución de las categorías de daño.

### 🛡️ 4.3. Estabilidad

La estabilidad del modelo se analizará comprobando si mantiene un desempeño similar al utilizar diferentes particiones de entrenamiento y validación.

Asimismo, se examinará la influencia del **desbalance entre las categorías de daño**, debido a que algunas clases pueden contener considerablemente más ejemplos que otras.

Se buscará evitar que imágenes o edificaciones demasiado similares aparezcan simultáneamente en los conjuntos de entrenamiento y prueba, con el propósito de obtener resultados más representativos de la capacidad de generalización.

Los resultados permitirán identificar las limitaciones del modelo y establecer en qué condiciones sus predicciones podrían utilizarse como apoyo para una evaluación preliminar posterior a un desastre.

<a id="entregas"></a>

## 📅 5. Entregas del curso

| Entrega | Objetivo | Estado |
|:--:|:--|:--:|
| **T1 · 30 %** | Problema, dataset, GitHub y marco VDS | 🟡 Documentación avanzada; exposición no realizada |
| **T2 · 30 %** | Análisis exploratorio de datos y plan de algoritmos | ⚪ Pendiente |
| **T3 · 40 %** | Implementación en Python, evaluación e informe final | ⚪ Pendiente |

Los estados se actualizarán conforme el grupo complete y verifique cada etapa.

<a id="colaboracion"></a>

## 🤝 6. Colaboración y uso de IA

El repositorio es el espacio compartido del **Grupo 04**. Cada integrante puede registrar su participación a través de *commits* identificables.

**Ejemplos de mensajes de commit:**

- `docs: mejorar el marco VDS`
- `data: describir el subconjunto seleccionado`
- `eda: incorporar gráficos de clases`
- `model: implementar clasificador base`
- `results: documentar métricas`

> [!CAUTION]
> El curso permite herramientas de IA como apoyo, pero requiere una **bitácora de prompts e iteraciones**, verificación de fuentes primarias y revisión responsable del código. Este README contiene redacción asistida por IA y deberá documentarse en la bitácora del equipo.

<a id="referencias"></a>

## 📚 7. Referencias

- Gupta, R., et al. (2019). *xBD: A Dataset for Assessing Building Damage from Satellite Imagery.* [arXiv:1911.09296](https://arxiv.org/abs/1911.09296).
- Yu, B., & Barter, R. (2024). *Veridical Data Science.* MIT Press. [Sitio del libro](https://vdsbook.com/).
- xView2. [Proyecto y dataset](https://www.xview2.org/).

---

<div align="center">
<sub>🛰️ Grupo 04 · Aplicaciones de IA en Estructuras · Repositorio académico en desarrollo</sub>
</div>
