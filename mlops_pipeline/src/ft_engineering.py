import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")


class ToCategory(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X


class OutliersToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, bounds=None):
        self.bounds = bounds or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].where(X[col].between(low, high), np.nan)
        return X


class KNNColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_neighbors=5):
        self.cols = cols or []
        self.n_neighbors = n_neighbors
        self._existing_cols = []
        self._scaler = StandardScaler()
        self._imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self._existing_cols = [col for col in self.cols if col in X.columns]
        if self._existing_cols:
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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if {"total_otros_prestamos", "capital_prestado", "salario_cliente"}.issubset(X.columns):
            X["relacion_deuda_ingreso"] = np.where(
                X["salario_cliente"] > 0,
                (X["total_otros_prestamos"] + X["capital_prestado"]) / X["salario_cliente"],
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
                ((X["cuota_pactada"] * X["plazo_meses"]) - X["capital_prestado"]) / X["capital_prestado"],
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
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.ct_ = None

    def fit(self, X, y=None):
        self.numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", ohe, self.categorical_features),
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

CATEGORY_COLUMNS = ["tipo_credito", "Pago_atiempo"]

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


pipeline_basemodel = Pipeline(
    steps=[
        ("drop_columns", ColumnDropper(cols_to_drop=COLUMNS_TO_DROP)),
        ("to_category", ToCategory(cols=CATEGORY_COLUMNS)),
        ("outliers_to_nan", OutliersToNaN(bounds=OUTLIER_BOUNDS)),
        ("imputation", KNNColumnImputer(cols=IMPUTE_COLUMNS, n_neighbors=5)),
        ("derived_features", DerivedFeatures()),
    ]
)


pipeline_ml = Pipeline(
    steps=[
        ("basemodel", pipeline_basemodel),
        ("preprocessor", AutoPreprocessorToDF()),
    ]
)


def split_features_target(df, target_col="Pago_atiempo"):
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
