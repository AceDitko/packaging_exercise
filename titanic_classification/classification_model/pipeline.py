# Categorical variable imports
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# Imputation imports
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

# Core imports
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer

price_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # add missing indicator to numerical vars
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars),
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_vars,
            ),
        ),
        # ==== VARIABLE TRANSFORMATION =====
        # Extract letter from cabin
        (
            "extract_cabin_letter",
            ExtractLetterTransformer(variables=config.model_config.cabin_vars),
        ),
        # == CATEGORICAL ENCODING
        # Remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05, n_categories=1, variables=config.model_config.categorical_vars
            ),
        ),
        # encode categorical using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True,
                variables=config.model_config.categorical_vars,
            ),
        ),
        # Scale
        ("scaler", StandardScaler()),
        ("Logit", LogisticRegression(C=0.0005, random_state=0)),
    ]
)
