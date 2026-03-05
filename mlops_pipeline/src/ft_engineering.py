import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops a configured list of columns from a DataFrame.

    This estimator is stateless other than remembering the configured column list.
    It is safe to include column names that do not exist in the input (they will
    be ignored).

    Args:
        cols_to_drop (Optional[List[str]]): List of column names to remove from the
            incoming DataFrame. If None or empty list no columns are dropped.

    Example:
        >>> tr = ColumnDropper(cols_to_drop=["a", "b"])
        >>> tr.fit(pd.DataFrame({"a":[1], "c":[2]})).transform(pd.DataFrame({"a":[1], "c":[2]}))
           c
        0  2
    """

    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")


class ToCategory(BaseEstimator, TransformerMixin):
    """Transformer that casts selected columns to pandas 'category' dtype.

    Useful to ensure deterministic behavior for downstream one-hot encoding and
    for preserving categorical semantics.

    Args:
        cols (Optional[List[str]]): List of column names to convert to category.
            Missing columns are ignored.

    Attributes:
        n_features_in_ (int): Number of features seen during fit.

    Example:
        >>> tr = ToCategory(cols=["tipo"])
        >>> tr.fit(pd.DataFrame({"tipo":["A", "B"], "x":[1,2]})).transform(pd.DataFrame({"tipo":["A","B"], "x":[1,2]})).dtypes["tipo"].name
        'category'
    """

    def __init__(self, cols=None):
        self.cols = cols or []

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X


class OutliersToNaN(BaseEstimator, TransformerMixin):
    """Transformer that replaces out-of-range values with NaN for specified columns.

    For each column provided in `bounds`, values outside the inclusive interval
    [low, high] are set to NaN. This is useful before imputing missing/outlier
    values.

    Args:
        bounds (Optional[Dict[str, Tuple[float, float]]]): Mapping from column name
            to (low, high) inclusive bounds. Columns not present in the DataFrame
            are ignored.

    Example:
        >>> tr = OutliersToNaN(bounds={"age": (18, 80)})
        >>> tr.fit(pd.DataFrame({"age":[20,90]})).transform(pd.DataFrame({"age":[20,90]}))
           age
        0   20.0
        1    NaN
    """

    def __init__(self, bounds=None):
        self.bounds = bounds or {}

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].where(X[col].between(low, high), np.nan)
        return X


class KNNColumnImputer(BaseEstimator, TransformerMixin):
    """Impute missing numeric values using KNN imputation on selected columns.

    The transformer standardizes the selected columns, fits a KNNImputer on the
    scaled data during fit, and applies the same scaling + imputation + inverse
    scaling at transform time.

    Args:
        cols (Optional[List[str]]): List of column names to impute. Columns that
            do not exist in the training DataFrame are ignored.
        n_neighbors (int): Number of neighbors to use for KNN imputation.

    Attributes:
        _existing_cols (List[str]): Subset of requested columns that exist in the fitted DataFrame.
        _scaler (StandardScaler): Fitted scaler for the selected columns.
        _imputer (KNNImputer): Fitted imputer instance.

    Example:
        >>> df = pd.DataFrame({"a":[1.0, np.nan, 3.0], "b":[1.0,2.0,3.0]})
        >>> tr = KNNColumnImputer(cols=["a"], n_neighbors=1)
        >>> tr.fit(df).transform(df)
             a    b
        0  1.0  1.0
        1  2.0  2.0
        2  3.0  3.0
    """

    def __init__(self, cols=None, n_neighbors=5):
        self.cols = cols or []
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self._existing_cols = [col for col in self.cols if col in X.columns]
        if self._existing_cols:
            self._scaler = StandardScaler()
            self._imputer = KNNImputer(n_neighbors=self.n_neighbors)
            scaled = self._scaler.fit_transform(X[self._existing_cols])
            self._imputer.fit(scaled)
        return self

    def transform(self, X):
        X = X.copy()
        if not self._existing_cols:
            return X

        scaled = self._scaler.transform(X[self._existing_cols])
        imputed_scaled = self._imputer.transform(scaled)
        imputed = self._scaler.inverse_transform(imputed_scaled)

        X[self._existing_cols] = pd.DataFrame(
            imputed,
            columns=self._existing_cols,
            index=X.index,
        )
        return X


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Create domain-specific derived features from existing columns.

    Adds several business-driven features if the required source columns exist:
      - relacion_deuda_ingreso: (total_otros_prestamos + capital_prestado) / salario_cliente
      - carga_pago_mensual: cuota_pactada / salario_cliente
      - ratio_interes_total: ((cuota_pactada * plazo_meses) - capital_prestado) / capital_prestado
      - grupo_edad_cliente: age bucketed into categories
      - cant_creditos_por_sector: sum of sector credit counts

    Args:
        None

    Example:
        >>> df = pd.DataFrame({"total_otros_prestamos":[0], "capital_prestado":[100], "salario_cliente":[200]})
        >>> DerivedFeatures().fit(df).transform(df)
           total_otros_prestamos  capital_prestado  salario_cliente  relacion_deuda_ingreso
        0                      0               100              200                     0.5
    """

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X.copy()

        if {"total_otros_prestamos", "capital_prestado", "salario_cliente"}.issubset(
            X.columns
        ):
            X["relacion_deuda_ingreso"] = np.where(
                X["salario_cliente"] > 0,
                (X["total_otros_prestamos"] + X["capital_prestado"])
                / X["salario_cliente"],
                0,
            )

        if {"cuota_pactada", "salario_cliente"}.issubset(X.columns):
            X["carga_pago_mensual"] = np.where(
                X["salario_cliente"] > 0,
                X["cuota_pactada"] / X["salario_cliente"],
                0,
            )

        if {"cuota_pactada", "plazo_meses", "capital_prestado"}.issubset(X.columns):
            X["ratio_interes_total"] = np.where(
                X["capital_prestado"] > 0,
                ((X["cuota_pactada"] * X["plazo_meses"]) - X["capital_prestado"])
                / X["capital_prestado"],
                0,
            )

        if "edad_cliente" in X.columns:
            X["grupo_edad_cliente"] = pd.cut(
                X["edad_cliente"],
                bins=[17, 30, 45, 60, 80],
                labels=["18-30", "31-45", "46-60", "61-80"],
                right=False,
            )

        if {
            "creditos_sectorFinanciero",
            "creditos_sectorCooperativo",
            "creditos_sectorReal",
        }.issubset(X.columns):
            X["cant_creditos_por_sector"] = (
                X["creditos_sectorFinanciero"]
                + X["creditos_sectorCooperativo"]
                + X["creditos_sectorReal"]
            )

        return X


class AutoPreprocessorToDF(BaseEstimator, TransformerMixin):
    """Preprocessor that scales numeric features and one-hot encodes categoricals, returning a DataFrame.

    On fit this transformer inspects the input DataFrame to determine numeric and
    categorical columns, builds a ColumnTransformer that standardizes numeric
    columns and one-hot-encodes categorical columns, and fits it. Transform returns
    a pandas DataFrame with column names obtained from the transformer.

    Attributes:
        numeric_features_ (List[str]): Numeric column names detected at fit time.
        categorical_features_ (List[str]): Non-numeric column names detected at fit time.
        ct_ (ColumnTransformer): Fitted ColumnTransformer instance.

    Example:
        >>> tr = AutoPreprocessorToDF()
        >>> tr.fit(pd.DataFrame({"a":[1,2], "b":["x","y"]})).transform(pd.DataFrame({"a":[3], "b":["z"]}))
             num__a  cat__b_z
        0  1.414214       1.0
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.numeric_features_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features_),
                ("cat", ohe, self.categorical_features_),
            ],
            remainder="drop",
        )
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        Xt = self.ct_.transform(X)
        feature_names = self.ct_.get_feature_names_out()
        return pd.DataFrame(Xt, columns=feature_names, index=X.index)


COLUMNS_TO_DROP = [
    "fecha_prestamo",
    "tendencia_ingresos",
    "promedio_ingresos_datacredito",
]

CATEGORY_COLUMNS = ["tipo_credito"]

OUTLIER_BOUNDS = {
    "edad_cliente": (18, 80),
    "cant_creditosvigentes": (0, 50),
    "plazo_meses": (0, 70),
}

IMPUTE_COLUMNS = [
    "saldo_mora_codeudor",
    "saldo_principal",
    "saldo_total",
    "saldo_mora",
    "edad_cliente",
    "puntaje_datacredito",
    "plazo_meses",
    "cant_creditosvigentes",
]

# 2. Pipeline Base

pipeline_basemodel = Pipeline(
    steps=[
        ("drop_columns", ColumnDropper(cols_to_drop=COLUMNS_TO_DROP)),
        ("to_category", ToCategory(cols=CATEGORY_COLUMNS)),
        ("outliers_to_nan", OutliersToNaN(bounds=OUTLIER_BOUNDS)),
        ("imputation", KNNColumnImputer(cols=IMPUTE_COLUMNS, n_neighbors=5)),
        ("derived_features", DerivedFeatures()),
    ]
)


# 3. Pipeline ML
def make_pipeline_ml() -> Pipeline:
    """Create and return a fresh, unfitted machine-learning pipeline.

    The returned pipeline composes the base feature engineering pipeline with
    an AutoPreprocessorToDF instance so that the output of fit/transform is a
    pandas DataFrame ready for model consumption.

    Returns:
        sklearn.pipeline.Pipeline: New pipeline instance combining base feature
            engineering steps and the DataFrame-returning preprocessor.

    Example:
        >>> p = make_pipeline_ml()
        >>> isinstance(p, Pipeline)
        True
    """
    base = Pipeline(
        steps=[
            ("drop_columns", ColumnDropper(cols_to_drop=COLUMNS_TO_DROP)),
            ("to_category", ToCategory(cols=CATEGORY_COLUMNS)),
            ("outliers_to_nan", OutliersToNaN(bounds=OUTLIER_BOUNDS)),
            ("imputation", KNNColumnImputer(cols=IMPUTE_COLUMNS, n_neighbors=5)),
            ("derived_features", DerivedFeatures()),
        ]
    )
    return Pipeline(
        steps=[
            ("basemodel", base),
            ("preprocessor", AutoPreprocessorToDF()),
        ]
    )


# Definir columnas numéricas y categóricas
pipeline_ml = make_pipeline_ml()


def split_features_target(df, target_col="Pago_atiempo"):
    """Split a DataFrame into feature matrix X and target series y.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and the target column.
        target_col (str): Name of the target column to separate. Defaults to "Pago_atiempo".

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Tuple with (X, y) where X is the DataFrame
            without the target column and y is the target Series.

    Raises:
        ValueError: If `target_col` is not present in `df`.

    Example:
        >>> df = pd.DataFrame({"Pago_atiempo":[1,0], "x":[10,20]})
        >>> X, y = split_features_target(df, "Pago_atiempo")
        >>> list(X.columns)
        ['x']
    """
    if target_col not in df.columns:
        raise ValueError(
            f"La columna objetivo '{target_col}' no existe en el dataframe."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
