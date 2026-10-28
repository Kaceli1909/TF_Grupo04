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
