# CC5514-Analisis-Topologico-de-Datos

## Filtración de Subniveles y Firmas Topológicas de Curvas Individuales

### Integrantes

* Paula Guerrero
* Pamela Mendoza
* Tomás Ubilla

### Descripción del Proyecto

Este proyecto estudia las Curvas de Andrews desde una perspectiva de análisis topológico de datos (TDA). La idea principal es representar cada observación del dataset como una función periódica definida sobre ($S^1$), para luego analizar su estructura mediante filtraciones de subniveles y homología persistente.

A partir de las curvas generadas, buscamos extraer firmas topológicas que permitan identificar patrones y diferencias entre clases de calidad del vino.

El trabajo contempla:

* Generación y preprocesamiento de Curvas de Andrews.
* Construcción de complejos cubicales periódicos en 1D.
* Cálculo de diagramas de persistencia ($H_0$, $H_1$).
* Extracción de estadísticas topológicas.
* Comparación de firmas topológicas entre distintas clases.
* Estudio de estabilidad frente a cambios de resolución y muestreo.

### Objetivo

El objetivo del proyecto es analizar si las características topológicas de las Curvas de Andrews contienen información discriminativa sobre la calidad del vino, utilizando herramientas de topología computacional y homología persistente.

### Dataset

El proyecto utiliza el dataset de calidad de vinos, considerando atributos fisicoquímicos como:

* fixed acidity
* volatile acidity
* citric acid
* residual sugar
* chlorides
* free sulfur dioxide
* total sulfur dioxide
* density
* pH
* sulphates
* alcohol

La variable objetivo corresponde a la calidad (`quality`) del vino.

### Estructura

* `andrews.py`
  * Generación y codificación de Curvas de Andrews.
* `sublevel_persistence.py`
  * Construcción de filtraciones y cálculo de persistencia.
