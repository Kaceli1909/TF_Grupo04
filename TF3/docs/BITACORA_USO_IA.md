# 🧾 Bitácora de uso de inteligencia artificial — TF3

**Proyecto:** Evaluación de daño visible en edificaciones mediante imágenes satelitales PRE/POST de México y Palu  
**Grupo:** 04 · Aplicaciones de IA en Estructuras  
**Herramienta consultada:** ChatGPT (OpenAI)  
**Alcance del registro:** consultas seleccionadas del desarrollo, depuración, análisis y documentación del TF3.

> **Criterio de registro.** Los prompts siguientes se **reescribieron con redacción técnica y orden cronológico aproximado** a partir de solicitudes realizadas durante el desarrollo. No son citas textuales ni una exportación íntegra del historial del chat. Se agrupan varias interacciones de seguimiento cuando abordan una misma tarea. Las métricas, conteos y errores se atribuyen a los registros de CIMLAB AI y archivos revisados, no a resultados experimentales generados por el modelo conversacional.

## Propósito y forma de uso

La herramienta de IA se utilizó como apoyo para **plantear alternativas metodológicas, revisar fallos de programación, explicar métricas, ayudar a redactar y organizar resultados y elaborar materiales de presentación**. El equipo proporcionó imágenes, archivos, capturas de errores y decisiones de alcance. Los entrenamientos y sus salidas se ejecutaron en el entorno local mediante CIMLAB AI.

### Criterios de trazabilidad

- **Prompt reformulado:** síntesis fiel del objetivo de una o varias consultas; no reproducción literal del mensaje original.
- **Asistencia obtenida:** propuesta, diagnóstico, explicación o material elaborado con apoyo de ChatGPT.
- **Evidencia y comprobación:** registro, captura o decisión del equipo que permitió aceptar, corregir o limitar la respuesta.
- **Pendientes:** no se consideran verificados los ensayos no finalizados, los resultados de `test` ni la generalización entre eventos.

## A. Enfoque, datos y diagnóstico inicial

### Consulta 01

**Prompt reformulado:** Retomar el desarrollo del sistema de IA para evaluar daños en edificaciones con imágenes satelitales y revisar la última versión disponible de la aplicación.

**Asistencia obtenida:** Revisión del flujo de trabajo y de la pestaña Auditoría; continuidad desde el experimento inicial con México.

**Comprobación / decisión del equipo:** Revisión de la interfaz y resultados por el equipo.

### Consulta 02

**Prompt reformulado:** Analizar por qué los modelos de la primera colección de México obtenían valores bajos de F1, recall y precisión, utilizando sus salidas de validación.

**Asistencia obtenida:** Identificación del desbalance de clases como posible factor; propuesta de auditoría visual.

**Comprobación / decisión del equipo:** Comparación con métricas reales de CIMLAB AI; no se atribuyó una causa única sin más pruebas.

### Consulta 03

**Prompt reformulado:** Examinar ejemplos de falsos positivos y falsos negativos, comparando las imágenes PRE y POST junto con el polígono del edificio.

**Asistencia obtenida:** Revisión de errores por edificio y propuesta de contrastar la calidad de los recortes.

**Comprobación / decisión del equipo:** El equipo compartió capturas y un informe de auditoría.

### Consulta 04

**Prompt reformulado:** Proponer mejoras de comparabilidad de las imágenes PRE/POST sin introducir detalles inexistentes ni alterar los datos originales.

**Asistencia obtenida:** Criterios para inspección de bordes, iluminación, nitidez y registro geométrico; distinción entre ayudas visuales y transformaciones de entrenamiento.

**Comprobación / decisión del equipo:** Se distinguieron las correcciones implementadas de las mejoras solo propuestas.

### Consulta 05

**Prompt reformulado:** Indicar cómo generar y reunir las auditorías de todos los modelos entrenados para comparar errores comunes y complementarios.

**Asistencia obtenida:** Procedimiento de exportación por modelo, lectura de FP/FN y criterios para comparar informes.

**Comprobación / decisión del equipo:** Exportaciones de validación y capturas revisadas durante el desarrollo.

## B. Adaptación del sistema a nuevos eventos y equipos

### Consulta 06

**Prompt reformulado:** Adaptar CIMLAB AI para que otros integrantes del grupo puedan ejecutarla en equipos con CPU o GPU, sin depender de las rutas de una sola computadora.

**Asistencia obtenida:** Diseño de instalación y actualización reutilizando dependencias ya existentes.

**Comprobación / decisión del equipo:** La aplicación se ejecutó en el equipo del usuario tras varias correcciones del actualizador.

### Consulta 07

**Prompt reformulado:** Permitir seleccionar imágenes TIFF y anotaciones JSON existentes en carpetas locales o descargarlas desde el repositorio del grupo.

**Asistencia obtenida:** Diseño de fuentes locales y descarga con Git LFS.

**Comprobación / decisión del equipo:** Se verificó posteriormente la detección y descarga de México y Palu.

### Consulta 08

**Prompt reformulado:** Hacer dinámica la detección de eventos disponibles en GitHub, para incorporar Palu u otros desastres sin codificar cada evento en la aplicación.

**Asistencia obtenida:** Actualización del catálogo de eventos mediante consulta al repositorio.

**Comprobación / decisión del equipo:** La interfaz mostró `mexico-earthquake` y `palu-tsunami`.

### Consulta 09

**Prompt reformulado:** Definir una tarea binaria común para todos los desastres compatibles y advertir cuando las etiquetas de un nuevo dataset no tengan equivalencia clara.

**Asistencia obtenida:** Reglas de mapeo y bloqueo/revisión de etiquetas desconocidas.

**Comprobación / decisión del equipo:** Se corrigió la definición de la clase positiva por decisión del usuario.

### Consulta 10

**Prompt reformulado:** Establecer la clasificación final como «no dañado» (0) frente a «dañado» (1), incluyendo daño menor, mayor y destrucción dentro de la clase positiva.

**Asistencia obtenida:** Mapeo explícito: `no-damage → 0`, `minor-damage/major-damage/destroyed → 1`.

**Comprobación / decisión del equipo:** La decisión de agrupar las intensidades fue indicada expresamente por el usuario.

### Consulta 11

**Prompt reformulado:** Preparar una actualización de CIMLAB AI que conserve las bibliotecas instaladas, los TIFF, JSON, recortes y modelos existentes.

**Asistencia obtenida:** Parche incremental y respaldo de archivos.

**Comprobación / decisión del equipo:** El usuario reportó problemas del instalador; se corrigió y volvió a ejecutar.

## C. Descarga, preparación y correcciones técnicas

### Consulta 12

**Prompt reformulado:** Verificar si el repositorio del grupo ya contiene los datos de Palu y si la estructura de archivos es utilizable junto con la de México.

**Asistencia obtenida:** Comprobación de carpetas, archivos PRE/POST y anotaciones JSON.

**Comprobación / decisión del equipo:** Se detectaron 196 pares de Palu y se confirmó la compatibilidad del esquema xBD.

### Consulta 13

**Prompt reformulado:** Revisar la descarga de archivos Git LFS y comprobar la integridad del conjunto de ambos eventos antes de preparar los recortes.

**Asistencia obtenida:** Interpretación del registro de descarga y auditoría de anotaciones.

**Comprobación / decisión del equipo:** El registro mostró 386 TIFF de México y 392 de Palu.

### Consulta 14

**Prompt reformulado:** Analizar la distribución de las cuatro etiquetas xBD y determinar cuántos edificios válidos y dañados se obtienen al combinar los dos eventos.

**Asistencia obtenida:** Resumen por evento, clase y exclusiones.

**Comprobación / decisión del equipo:** El registro registró 106 540 edificios válidos, 8660 dañados y 722 `un-classified` excluidos.

### Consulta 15

**Prompt reformulado:** Recomendar una preparación que conserve los edificios sin daño, evite sobrescribir los experimentos anteriores y documente la cantidad de recortes.

**Asistencia obtenida:** Configuración de trabajo separada y máximo de negativos en cero (sin limitación).

**Comprobación / decisión del equipo:** El usuario inició la preparación conjunta con 385 escenas.

### Consulta 16

**Prompt reformulado:** Diagnosticar por qué una versión del programa recorría todas las escenas sin generar recortes y mostrar el error específico en vez de una alerta genérica.

**Asistencia obtenida:** Parche de diagnóstico que dejó visible el error de lectura TIFF.

**Comprobación / decisión del equipo:** Capturas del error `UnidentifiedImageError` proporcionadas por el usuario.

### Consulta 17

**Prompt reformulado:** Investigar el formato real de los TIFF y proponer una lectura segura de imágenes RGB almacenadas como `int16`.

**Asistencia obtenida:** Conversión exacta a `uint8` solo cuando los valores de cada imagen están entre 0 y 255; rechazo de rangos incompatibles.

**Comprobación / decisión del equipo:** El diagnóstico posterior confirmó la lectura correcta de seis muestras, no de la totalidad de archivos individualmente.

### Consulta 18

**Prompt reformulado:** Evitar franjas negras generadas por los recortes cuando la ventana PRE/POST rebasa los límites de la imagen.

**Asistencia obtenida:** Relleno por reflexión en bordes y exclusión de ventanas predominantemente externas.

**Comprobación / decisión del equipo:** La metodología se documentó como reducción de artefactos, no recuperación de información ausente.

### Consulta 19

**Prompt reformulado:** Comprobar el manifiesto conjunto y corregir las particiones de datos sin reprocesar las 385 escenas ni modificar los JPG existentes.

**Asistencia obtenida:** Separación original `tier1 → train`, `hold → val`, `test → test` y generación de respaldo del manifiesto.

**Comprobación / decisión del equipo:** El registro de corrección confirmó 63 189/19 537/23 814 edificios en train/val/test.

### Consulta 20

**Prompt reformulado:** Explicar por qué los resultados cambian cuando se vuelve a preparar el dataset o se modifica el máximo de edificios no dañados.

**Asistencia obtenida:** Interpretación de distribución de clases, semilla, particiones y variabilidad del entrenamiento.

**Comprobación / decisión del equipo:** Se recomendó registrar configuración y conservar versiones de experimento.

## D. Selección, entrenamiento y comparación de modelos

### Consulta 21

**Prompt reformulado:** Comparar Random Forest, DINOv2 + XGBoost, modelos siameses y arquitecturas profundas para elegir experimentos razonables bajo restricciones de tiempo y hardware.

**Asistencia obtenida:** Priorización de modelos distintos en enfoque y complejidad.

**Comprobación / decisión del equipo:** La selección final de ejecuciones fue realizada por el equipo.

### Consulta 22

**Prompt reformulado:** Interpretar el resultado obtenido al entrenar Random Forest con el dataset conjunto y compararlo con la referencia preliminar de México.

**Asistencia obtenida:** Análisis de AP, F1, recall y precisión; advertencia de que los conjuntos comparados no eran idénticos.

**Comprobación / decisión del equipo:** Valores obtenidos de la captura de CIMLAB AI, no generados por el asistente.

### Consulta 23

**Prompt reformulado:** Interpretar las métricas de DINOv2 + XGBoost y del ensamble Hybrid en validación y explicar la función de cada arquitectura.

**Asistencia obtenida:** Comparación numérica y descripción de Hybrid como combinación automática de probabilidades.

**Comprobación / decisión del equipo:** La composición exacta del ensamble quedó pendiente de corroboración mediante su archivo de configuración.

### Consulta 24

**Prompt reformulado:** Revisar la configuración de ResNet18 Early Fusion con 20 épocas y batch size 8 y explicar el tiempo de entrenamiento en CPU/GPU.

**Asistencia obtenida:** Interpretación de lotes por época, consumo computacional y early stopping.

**Comprobación / decisión del equipo:** Se registraron solo métricas intermedias de las épocas 1 y 2; no se presentaron como modelo finalizado.

### Consulta 25

**Prompt reformulado:** Incorporar ProtoNet DINO a la comparación final y actualizar los resultados del ensamble Hybrid.

**Asistencia obtenida:** Actualización de AP, F1, recall y precisión para cuatro modelos.

**Comprobación / decisión del equipo:** Datos transcritos de la pantalla de CIMLAB AI: ProtoNet y Hybrid actualizado.

### Consulta 26

**Prompt reformulado:** Elaborar matrices de confusión binarias para los modelos completados, explicando TP, FP, FN y TN.

**Asistencia obtenida:** Tablas y figuras comparativas de errores.

**Comprobación / decisión del equipo:** ProtoNet aportó conteos registrados; los demás se reconstruyeron desde métricas redondeadas y se identificaron como tales.

## E. Redacción académica, visualización y presentación

### Consulta 27

**Prompt reformulado:** Documentar la entrega TF3 en una carpeta independiente, manteniendo intactos el README y los materiales de TF2.

**Asistencia obtenida:** Estructura `TF3/` con README, scripts Python, artículo IEEE y resultados.

**Comprobación / decisión del equipo:** El usuario eligió subir los archivos manualmente a GitHub.

### Consulta 28

**Prompt reformulado:** Redactar y organizar el README y el artículo IEEE con el procedimiento de procesamiento TIFF, las particiones, los modelos y las métricas verificadas.

**Asistencia obtenida:** Redacción técnica, tablas, discusión, limitaciones y referencias.

**Comprobación / decisión del equipo:** Se evitó presentar validación como prueba final.

### Consulta 29

**Prompt reformulado:** Preparar una exposición con tablas, gráficos, matrices de confusión y explicación esquemática de Hybrid.

**Asistencia obtenida:** Presentación del método y resultados en diapositivas.

**Comprobación / decisión del equipo:** Se reconstruyó el PPTX después de un error de compatibilidad; no constituye dato experimental.

### Consulta 30

**Prompt reformulado:** Proponer conclusiones y futuras líneas de investigación: evaluación por evento, más arquitecturas y clasificación por intensidad de daño.

**Asistencia obtenida:** Discusión de límites del modelo binario y posible clasificación multiclase.

**Comprobación / decisión del equipo:** Se distinguieron propuestas futuras de experimentos ejecutados.

### Consulta 31

**Prompt reformulado:** Preparar un guion privado para cuatro expositores, además de preguntas y respuestas posibles para la defensa.

**Asistencia obtenida:** Material de práctica separado de los entregables públicos.

**Comprobación / decisión del equipo:** El guion se entregó fuera de la carpeta pública de GitHub.

### Consulta 32

**Prompt reformulado:** Preparar una bitácora de uso de ChatGPT con las solicitudes del proyecto organizadas y mejor redactadas, integrándola a README y artículo IEEE.

**Asistencia obtenida:** Este registro de consultas y apoyos de IA.

**Comprobación / decisión del equipo:** Los prompts se presentan como formulaciones normalizadas, no como transcripciones literales.

## Síntesis de la contribución y responsabilidad del equipo

| Actividad | Apoyo de ChatGPT | Responsabilidad y evidencia del equipo |
|:--|:--|:--|
| Diseño del enfoque | Análisis de opciones y redacción de procedimientos | Definición de tarea binaria; elección de eventos y arquitecturas |
| Preparación técnica | Diagnósticos y propuestas de parches para TIFF, bordes y particiones | Ejecución local, capturas de errores, verificación de recortes y manifiesto |
| Entrenamiento | Explicación de parámetros y métricas | Entrenamiento en CIMLAB AI; registros de validación de Random Forest, DINOv2, ProtoNet y Hybrid |
| Resultados | Estructuración de tablas y figuras | Contraste con capturas y distinción entre conteos registrados y reconstruidos |
| Informe y presentación | Apoyo editorial, tablas, gráficos y guion | Selección, revisión y presentación del contenido |

## Limitaciones del registro y uso responsable

1. La lista reúne **solicitudes representativas**, agrupadas y mejor redactadas. No permite reconstruir cada mensaje, cada respuesta ni la fecha exacta de todas las interacciones.
2. La asistencia mediante IA no sustituye la revisión del código, la integridad del dataset, la validación de los resultados ni las referencias bibliográficas de fuentes originales.
3. Los resultados publicados corresponden a **validación**; `test` se mantuvo reservado. ResNet18 Early Fusion quedó sin entrenamiento concluido en el corte reportado.
4. El formato de salida de ChatGPT se revisó y, cuando surgieron errores (actualizadores, lectura TIFF o presentación), se corrigió mediante iteraciones posteriores. La presencia de una respuesta previa no se considera prueba de que fuera correcta.
5. No se atribuye a la IA la ejecución de los entrenamientos locales ni la autoría de las imágenes satelitales o etiquetas del conjunto xBD.

## Materiales del repositorio relacionados

- [README de TF3](../README.md): objetivo, preparación, modelos y resultados.
- [Artículo técnico](ARTICULO_IEEE_BORRADOR.md) y [fuente IEEE en LaTeX](ARTICULO_IEEE_PLANTILLA.tex): metodología y discusión.
- [Registro de experimentos](../resultados/REGISTRO_EXPERIMENTOS.md): métricas de validación observadas.
- [Guía de reproducibilidad](GUIA_REPRODUCIBILIDAD.md): dependencias, procesamiento y particiones.
- [Código Python](../codigo/): herramientas de auditoría y aplicación.

*Documento de transparencia metodológica. Las formulaciones de prompts fueron normalizadas para facilitar su lectura académica.*
