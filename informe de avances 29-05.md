# Informe de avances 29-05 - Anna

## Resumen ejecutivo
Durante la jornada de trabajo del 29-05 se evaluaron dos enfoques principales para la detecciÃ³n de texto en imÃ¡genes mÃ©dicas:

1. OCR directo sobre imagen completa (con mÃºltiples preprocesamientos).
2. OCR restringido a regiones recortadas (bounding boxes provenientes de labels).

Luego de las pruebas y anÃ¡lisis de resultados, se concluyÃ³ que el enfoque de OCR sobre imagen completa ofrece mejor rendimiento prÃ¡ctico y mayor robustez general para este desafÃ­o.

## Hallazgos principales

### 1. El OCR sin bounding box previa mostrÃ³ mejor lectura general
En las pruebas realizadas, ejecutar OCR sobre la imagen completa permitiÃ³ recuperar mÃ¡s palabras Ãºtiles y con mayor estabilidad que recortar zonas especÃ­ficas antes de leer.

Esto se explica por varios factores:

- El motor OCR aprovecha mejor el contexto visual y estructural cuando ve la imagen completa.
- Los recortes exactos pueden cortar bordes de caracteres o perder informaciÃ³n relevante del entorno.
- En regiones pequeÃ±as, ciertos preprocesamientos pueden amplificar ruido y degradar la lectura.
- Si la caja no es casi perfecta, el OCR parte de una entrada ya limitada y con riesgo de error acumulado.

### 2. Dar labels/bounding boxes no mejorÃ³ el resultado final del OCR
Aunque conceptualmente parecerÃ­a que acotar el Ã¡rea deberÃ­a ayudar, en esta implementaciÃ³n no produjo mejoras consistentes en calidad de lectura.  
Por el contrario, agregÃ³ una dependencia crÃ­tica: la precisiÃ³n de la detecciÃ³n previa.

En otras palabras, se incorpora una nueva fuente de error:

- Error de detecciÃ³n/localizaciÃ³n de caja.
- Error de lectura OCR sobre una regiÃ³n potencialmente recortada de forma subÃ³ptima.

Esto aumenta la complejidad del pipeline sin un beneficio claro en rendimiento real.

## Impacto sobre el objetivo inicial del desafÃ­o
El objetivo inicial planteaba entrenar un modelo que entregara bounding boxes para luego aplicar OCR.  
Sin embargo, la evidencia obtenida indica que, para este caso, ese paso no mejora la lectura y puede perjudicarla si el detector no es casi perfecto.

Por ese motivo, se decidiÃ³ no continuar con entrenamiento de YOLO para esta etapa, ya que implicarÃ­a sumar complejidad y riesgo de mala detecciÃ³n en lugar de mejorar el anÃ¡lisis.

## Estrategia adoptada
Se optÃ³ por un enfoque centrado en:

1. OCR sobre imagen completa.
2. Uso de mÃºltiples variantes de preprocesamiento de imagen para maximizar recuperaciÃ³n de texto.
3. AplicaciÃ³n de reglas de validaciÃ³n y filtrado para consolidar resultados confiables.
4. IdentificaciÃ³n final de palabras candidatas a anonimizaciÃ³n.

Este enfoque resultÃ³ mÃ¡s consistente, mÃ¡s simple de mantener y mÃ¡s alineado con el rendimiento observado durante las pruebas.

## ConclusiÃ³n
La decisiÃ³n tÃ©cnica final es priorizar un pipeline de OCR full-image con preprocesamientos y reglas de consolidaciÃ³n, en lugar de incorporar una etapa previa de detecciÃ³n por bounding boxes con YOLO.  
Con la evidencia actual, esta estrategia ofrece mejor balance entre calidad de lectura, robustez y complejidad operativa para el proceso de anonimizaciÃ³n.

---

# INFORME DE PROGRESO
## Hackathon Treelogic
## Desidentificación Visual de Imágenes Médicas
### 29 de mayo de 2025

## 1. Contexto del Reto
El reto propuesto por Treelogic consiste en diseñar una solución que permita eliminar o anonimizar información identificable en imágenes médicas. El objetivo es proteger la privacidad de los pacientes facilitando al mismo tiempo el uso seguro de datos médicos para investigación y diagnóstico.

Las imágenes médicas contienen dos tipos de información identificable:

- Metadatos DICOM: nombre del paciente, fecha de nacimiento, ID hospitalario y otros campos embebidos en la cabecera del fichero.
- Texto incrustado visualmente (burned-in text): texto sobreimpreso directamente sobre los píxeles de la imagen, común en radiografías, ecografías y TACs.

## 2. Dataset Proporcionado
La organización ha facilitado un dataset etiquetado con anotaciones en formato YOLO estándar. El conjunto de datos contiene las siguientes características:

| Característica | Detalle |
|---|---|
| Tipo de imágenes | Radiografías médicas (.png) |
| Total de imágenes | ~400 |
| Conjunto de entrenamiento | ~320 imágenes (80%) |
| Conjunto de validación | ~80 imágenes (20%) |
| Formato de anotación | YOLO (bounding boxes normalizadas) |
| Número de clases | 5 |

Las cinco clases de información sensible etiquetadas son:

| ID | Clase | ¿Anonimizar? |
|---|---|---|
| 0 | name | Siempre |
| 1 | id | Siempre |
| 2 | age | Siempre |
| 3 | date | Depende del contexto (fecha adquisición vs. fecha nacimiento) |
| 4 | time | Depende del contexto |

## 3. Arquitectura de la Solución
La solución se estructura en dos capas complementarias:

### 3.1 Capa de Detección — Modelo YOLO
Se ha optado por el uso de YOLOv8 (You Only Look Once), un modelo de detección de objetos basado en redes neuronales convolucionales. La decisión se justifica por:

- Salida directa de bounding boxes con coordenadas precisas, eliminando la necesidad de postprocesado complejo.
- Transfer learning: el modelo parte de pesos preentrenados en ImageNet (`yolov8n.pt`), lo que permite obtener buenos resultados con tan solo 320 imágenes de entrenamiento.
- Clasificación por tipo de PII: el modelo distingue entre las 5 clases definidas, permitiendo aplicar lógica diferenciada para fechas y horas.

### 3.2 Capa de OCR — Pipeline Genérico
En paralelo, se ha desarrollado un pipeline de OCR (Reconocimiento Óptico de Caracteres) basado en Tesseract y OpenCV como solución de respaldo y para la detección de texto en imágenes sin etiquetar. Las características principales son:

- Múltiples pasadas de OCR con diferentes tratamientos de imagen (escala de grises, invertido, nitidez aumentada, escalado).
- Detección de bandas de color por saturación HSV para localizar texto sobre fondos de color (ej. texto blanco sobre fondo azul).
- Filtrado por confianza e IoU (Intersección sobre Unión) para eliminar detecciones falsas.
- Redacción mediante rectángulos negros rellenos sobre las regiones detectadas.

## 4. Progreso del Día
### 4.1 Pipeline de OCR
Se desarrolló e iteró un pipeline genérico de extracción de texto capaz de detectar texto en distintos colores y fondos sin configuración específica por color. Partiendo de una implementación básica con `pytesseract`, se incorporaron:

- Detección por pasadas múltiples con imágenes transformadas.
- Segmentación automática de bandas de color mediante análisis de saturación por fila.
- Filtrado de detecciones basura mediante expresiones regulares.
- Obtención de coordenadas de bounding box para cada token detectado.

### 4.2 Entrenamiento YOLO
Se configuró y lanzó el entrenamiento de YOLOv8n con el dataset proporcionado. Los pasos realizados fueron:

- Descarga manual de los pesos preentrenados `yolov8n.pt`.
- Corrección del fichero `data.yaml` (eliminación de caracteres inválidos y adición del campo `path`).
- Configuración del script de entrenamiento con aumentación de datos (flip, rotación, escala, mosaico).
- Lanzamiento del entrenamiento en CPU (Apple M5 Pro), 50 épocas, batch size 8.

### 4.3 Visualizador de Etiquetas
Se implementó una herramienta de visualización que superpone las anotaciones YOLO sobre las radiografías originales, con código de colores por clase, para verificar la calidad del dataset antes de confiar en los resultados del entrenamiento.

## 5. Próximos Pasos
- Evaluar las métricas del modelo entrenado (mAP50, precisión, recall) y ajustar hiperparámetros si es necesario.
- Implementar el script de redacción final: aplicar las bounding boxes detectadas por YOLO para ennegrecer las regiones de PII en nuevas imágenes.
- Integrar el pipeline completo en una API REST con FastAPI.
- Añadir una interfaz web básica con vista previa antes/después de la anonimización.
- Preparar la demo para la presentación ante el jurado.

## 6. Stack Tecnológico
| Componente | Tecnología |
|---|---|
| Detección de PII | YOLOv8 (Ultralytics) |
| OCR de respaldo | Tesseract + EasyOCR |
| Procesamiento de imagen | OpenCV, Pillow |
| Gestión de DICOM | pydicom |
| API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |
| Lenguaje | Python 3.9+ |
