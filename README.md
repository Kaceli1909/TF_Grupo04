# Evaluación Rapida de Daño Estructural Post- Sismo con imágenes satelitales
> Proyecto grupal – Entrega T1 
> Curso: _Aplicaciones de IA en Estructuras_ · Docente: _Kurt Walter Soncco Sinchi_ 
## 1. Problema de ingeniería estructural

Después de un sismo severo, las entidades de respuesta necesitan saber **qué edificaciones están dañadas y con qué severidad** para priorizar rescate, evacuación, inspecciones detalladas y habilitación de albergues. Hoy esa evaluación se hace principalmente con inspecciones de campo, que son lentas, peligrosas y requieren muchos especialistas en las primeras 24–48 horas.

Lima Metropolitana está en una zona de alta sismicidad y concentra una gran cantidad de viviendas autoconstruidas, por lo que un sismo mayor generaría miles de edificaciones por evaluar a la vez.

**Pregunta de investigación:**
¿Es posible clasificar de forma confiable el **nivel de daño estructural** de cada edificación (sin daño, daño menor, daño mayor, destruida) a partir de pares de imágenes satelitales **pre y post evento**, de manera que sirva como **triaje inicial** antes de la inspección de campo?
## 2. Dataset: xBD (xView2)

- **Fuente:** [xView2 / xBD](https://www.xview2.org/dataset) – Gupta et al. (2019), *xBD: A Dataset for Assessing Building Damage from Satellite Imagery*, [arXiv:1911.09296](https://arxiv.org/abs/1911.09296).
- **Contenido:** imágenes satelitales RGB de alta resolución (< 0.8 m/píxel) antes y después de 19 desastres naturales, con polígonos de edificaciones etiquetados según la *Joint Damage Scale* (0–3).
- **Tamaño:** ~22 000 imágenes y ~850 000 polígonos de edificaciones.
- **Evento sísmico de interés:** Sismo de Ciudad de México (19/09/2017).

## 3. Marco VDS – Principios PCS

El proyecto considera los principios del marco Veridical Data Science (VDS), desarrollado por Bin Yu y colaboradores, mediante el enfoque PCS: Predictibilidad, Computabilidad y Estabilidad.

### 3.1. Predictibilidad

El dataset xBD contiene imágenes satelitales de edificaciones antes y después de diferentes desastres naturales, acompañadas de etiquetas que describen sus niveles de daño visible.

La disponibilidad de imágenes y clasificaciones previamente establecidas permite plantear un problema de aprendizaje supervisado, en el que los algoritmos identifiquen patrones asociados con las diferentes categorías de daño.

Se espera analizar la capacidad predictiva de los modelos mediante métricas de clasificación, prestando especial atención a la identificación de edificaciones con daños mayores y destruidas.

### 3.2. Computabilidad

El conjunto de datos xBD dispone de imágenes satelitales y archivos de anotaciones que pueden procesarse mediante herramientas de Python.

Para el desarrollo del proyecto se prevé utilizar bibliotecas como NumPy, Pandas, Matplotlib, PyTorch y herramientas de procesamiento de imágenes.
Debido al tamaño del dataset y a los recursos computacionales que requiere el entrenamiento de redes neuronales, inicialmente se propone trabajar con una muestra representativa de imágenes correspondientes al terremoto de México de 2017.

La cantidad definitiva de imágenes se establecerá después de verificar su disponibilidad y la distribución de las categorías de daño.

### 3.3. Estabilidad

La estabilidad del modelo se analizará comprobando si mantiene un desempeño similar al utilizar diferentes particiones de entrenamiento y validación.

Asimismo, se examinará la influencia del desbalance entre las categorías de daño, debido a que algunas clases pueden contener considerablemente más ejemplos que otras.

Se buscará evitar que imágenes o edificaciones demasiado similares aparezcan simultáneamente en los conjuntos de entrenamiento y prueba, con el propósito de obtener resultados más representativos de la capacidad de generalización.

Los resultados permitirán identificar las limitaciones del modelo y establecer en qué condiciones sus predicciones podrían utilizarse como apoyo para una evaluación preliminar posterior a un desastre.
