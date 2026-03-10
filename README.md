📊 # **Telecom Churn Prediction: Análisis y Modelado Proactivo**

Este proyecto fue desarrollado como parte del Challenge Telecom X, con el objetivo de identificar patrones de abandono y construir un sistema de alerta temprana para la retención de clientes.

**1. Propósito del Análisis**
El objetivo principal es predecir la cancelación (Churn) de clientes utilizando técnicas de Machine Learning. Buscamos entender por qué los usuarios abandonan el servicio y proporcionar a la toma de decisiones empresarial una lista priorizada de clientes en riesgo, permitiendo acciones de fidelización preventivas antes de que la pérdida ocurra.

**2. Estructura del Proyecto**
El repositorio está organizado de la siguiente manera:

telecom_churn_analysis.ipynb: Cuaderno principal en Google Colab con todo el pipeline (EDA, Preprocesamiento, Modelado).

data/:

datos_tratados.csv: Dataset original tras limpieza inicial.

csv_limpio_para_ml.csv: Dataset final con ingeniería de variables y codificación listo para modelos.

visualizaciones/: Carpeta con los gráficos clave (Heatmap, Boxplots, Feature Importance).

README.md: Documentación del proyecto.

**3. Preparación de los Datos y Modelización**
Clasificación de Variables
Numéricas: customer_tenure (antigüedad), account_Charges.Monthly (cargos mensuales), entre otras.

Categóricas: account_Contract (tipo de contrato), internet_InternetService (tipo de internet), payment_method, etc.

Ingeniería y Transformación
Codificación: Aplicamos One-Hot Encoding para transformar variables categóricas en binarias, permitiendo que algoritmos como Random Forest y XGBoost las procesen matemáticamente.

Balanceo (SMOTE): Debido a que solo el 27% de los clientes cancelaban (clases desbalanceadas), utilizamos SMOTE para generar ejemplos sintéticos de la clase minoritaria, logrando un equilibrio 50/50 en el entrenamiento.

División de Datos: Separamos el dataset en 80% Entrenamiento y 20% Prueba utilizando una división estratificada para mantener la representatividad de la fuga en ambos conjuntos.

Justificación de Modelos
Random Forest: Seleccionado por su robustez y precisión (85%) al manejar datos no lineales.

XGBoost: Utilizado para maximizar el Recall (87%), asegurando que detectemos a la mayor cantidad de desertores posibles.

Árbol de Decisión: Empleado con fines de interpretabilidad para explicar las reglas de negocio a la junta directiva.

**4. Insights del Análisis Exploratorio (EDA)**
Durante la fase de exploración, descubrimos tres hallazgos críticos:

El Factor Contrato: Los clientes con contratos "Mes a Mes" representan el mayor volumen de fugas.

El Umbral de Precio: Existe un punto de quiebre en los $70 USD. Por encima de este valor, la densidad de cancelación aumenta drásticamente.

Fidelidad Temprana: La mayor probabilidad de fuga ocurre en los primeros 12 meses. Si un cliente supera los 24 meses, su lealtad aumenta exponencialmente.

**5. Instrucciones de Ejecución*
Requisitos Previos
Es necesario instalar las siguientes librerías de Python:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
Ejecución
Clona este repositorio.

Carga el archivo csv_limpio_para_ml.csv en tu entorno de Google Colab o Jupyter.

Asegúrate de actualizar la ruta de carga de datos en la primera celda del cuaderno:

Python
df = pd.read_csv('csv_limpio_para_ml.csv')
Ejecuta las celdas en orden secuencial para reproducir los resultados de la matriz de confusión y el reporte de métricas.
