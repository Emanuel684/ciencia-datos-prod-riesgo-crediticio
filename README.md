<!-- ...existing code... -->

# Ciencia de Datos en Producción — Entregable 2

Resumen del proyecto para el curso "Ciencia de Datos en Producción". Este repositorio contiene el flujo completo de preparación de datos, ingeniería de features, entrenamiento, evaluación y persistencia de modelos para un problema de riesgo crediticio (predicción de Pago_atiempo).

## Objetivo
Construir, evaluar y seleccionar modelos que predigan si un cliente pagará a tiempo (columna `Pago_atiempo`) a partir de datos de crédito. Entregar una tubería reproducible y segura para entrenamiento y evaluación, con énfasis en buen manejo de preprocesamiento, desbalance de clases y validación correcta.

## Datos
- Archivo principal: `Base_de_datos.xlsx`
- Variables clave (no exhaustivo): `tipo_credito`, `capital_prestado`, `plazo_meses`, `edad_cliente`, `tipo_laboral`, `salario_cliente`, `total_otros_prestamos`, `cuota_pactada`, `puntaje_datacredito`, `saldo_mora`, `Pago_atiempo`.
- Nota: el notebook `comprension_eda.ipynb` contiene el análisis exploratorio y las decisiones de limpieza/derivación de variables.

## Estructura del repositorio
- mlops_pipeline/
  - src/
    - ft_engineering.py      -> Transformaciones y pipelines de features
    - model_training.py      -> Construcción, evaluación y selección de modelos
    - Cargar_datos.ipynb     -> (celda vacía / auxiliar)
    - comprension_eda.ipynb  -> EDA y limpieza exploratoria
  - artifacts/               -> Salida (resumen, gráfico, modelo seleccionado)
- Base_de_datos.xlsx        -> Dataset (entrada)
- README.md                 -> Este archivo

## Requisitos e instalación (Windows)
1. Crear/activar entorno (recomendado):
   - python -m venv .venv
   - .\.venv\Scripts\activate
2. Instalar dependencias:
   - python -m pip install -r requirements.txt
   - Si no hay `requirements.txt`: python -m pip install pandas numpy scikit-learn matplotlib joblib openpyxl
3. Ejecutar scripts desde la raíz del repo:
   - Entrenar y seleccionar modelo:
     - python .\mlops_pipeline\src\model_training.py
     (o ejecutar desde un IDE/Notebooks)

## Cómo usar
1. Colocar `Base_de_datos.xlsx` en la raíz del proyecto (o indicar otra ruta al `train_and_select_model`).
2. Ejecutar el script de entrenamiento para generar:
   - `artifacts/model_summary.csv`
   - `artifacts/model_comparison.png`
   - `artifacts/best_model.joblib`
3. Para evaluar localmente, abrir `comprension_eda.ipynb` o ejecutar funciones desde `model_training.py`.

## Diseño de la tubería de features (ft_engineering.py)
- Pasos principales implementados:
  - ColumnDropper: eliminar columnas irrelevantes.
  - ToCategory: forzar columnas específicas como categoricas.
  - OutliersToNaN: convertir valores fuera de rango a NaN antes de imputación.
  - KNNColumnImputer: imputación KNN sobre columnas numéricas con escalado.
  - DerivedFeatures: creación de variables como `relacion_deuda_ingreso`, `carga_pago_mensual`, `ratio_interes_total`, `grupo_edad_cliente`, `cant_creditos_por_sector`.
  - AutoPreprocessorToDF: estandariza númericos y one-hot-encodes categoricas, devolviendo DataFrame.

## Entrenamiento y evaluación (model_training.py)
- Candidate models: HeuristicModel, LogisticRegression (class_weight="balanced"), RandomForestClassifier (class_weight="balanced").
- Proceso:
  1. Leer datos y separar features/target (`split_features_target`).
  2. Train/test split estratificado (test_size=0.25).
  3. Cross-validation (KFold, n_splits=10) sobre el conjunto de entrenamiento con `cross_validate`.
  4. Entrenar sobre todo el train y evaluar sobre test.
  5. Seleccionar mejor modelo usando una métrica compuesta (test F1, estabilidad CV, tiempo de fit).
  6. Guardar artefactos y el pipeline final.

## Por qué pudo obtenerse "score perfecto" y cómo evitarlo
Puntajes perfectos casi siempre indican:
- Fuga de datos (data leakage): alguna característica contiene información directa del target (o se genera a partir del target).
- Evaluación incorrecta: usar transformaciones calculadas con todo el dataset antes del split o aplicar balanceo / imputación antes de separar train/test o durante CV sin una pipeline correctamente encapsulada.
- Overfitting extremo: modelo memoriza patrones no generalizables (p. ej. features con alta cardinalidad que coinciden con la etiqueta).

Recomendaciones prácticas:
- Asegurar que todo preprocesamiento que aprende de los datos (scaler, imputer, encoder) esté dentro de una Pipeline y que cross_validate / GridSearch use esa Pipeline: evita fit antes del split.
- No balancear (bootstrap, undersample/oversample) antes del train/test split. Si balanceas, hacerlo únicamente dentro del proceso de entrenamiento, idealmente con pasos que soporten fit_resample (p. ej. imbalanced-learn Pipelines: `imblearn.pipeline.Pipeline`) o aplicarlo solo sobre el set de entrenamiento antes de entrenar el estimador final.
- Revisar las features derivadas: comprobar correlación directa entre features y target; eliminar o modificar las que causen fuga.
- Usar métricas robustas para datos desbalanceados: balanced_accuracy, recall, F1 por clase y curva ROC-AUC; la exactitud (accuracy) puede ser engañosa.
- Validación: usar CV estratificado (StratifiedKFold) para clasificación desbalanceada, revisar varianza entre folds.

## Pruebas y diagnóstico
- Añadir tests que:
  - Verifiquen que la pipeline no filtra información del test al train (pruebas con datos sintéticos).
  - Confirmen que `cross_validate` y `fit` usan pipelines independientes (sin compartir estado).
  - Detecten si alguna columna está altamente correlacionada con el target (posible fuga).
- Ejecutar pytest:
  - pytest -q

## Reproducibilidad
- Fijar seeds: `RANDOM_STATE = 42` ya usado en el proyecto.
- Registrar versiones de paquetes (pip freeze > requirements.txt).

## Buenas prácticas sugeridas
- Encapsular todo aprendizaje (scaler, imputer, encoder) en sklearn Pipeline u objetos compatibles (imblearn para resampling).
- Probar con datos sintéticos ruidosos para validar que el proceso de evaluación produce resultados razonables (no perfectos).
- Auditar correlaciones entre todas las features y el target antes del entrenamiento.
- Documentar cualquier limpieza o transformación que utilice la variable objetivo (evitarlo en la mayoría de los casos).

## Artefactos generados
- artifacts/model_summary.csv — tabla con métricas por candidato.
- artifacts/model_comparison.png — gráfico comparativo.
- artifacts/best_model.joblib — pipeline final persistida.

## Contacto
- Autor: Emanuel Acevedo Muñoz
- Curso: Ciencia de Datos en Producción — Entregable 2


comprension_eda > model_training -> model_evaluation -> model_deploy -> model_monitoring
