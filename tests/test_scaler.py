import numpy as np
import pandas as pd
from src.features import FeatureEngineer

def test_scaler_no_lookahead():
    fe = FeatureEngineer()

    # Datos de entrenamiento
    X_train = pd.DataFrame(np.random.randn(100, len(fe.feature_columns)),
                           columns=fe.feature_columns)

    # Datos de validación
    X_valid = pd.DataFrame(np.random.randn(20, len(fe.feature_columns)),
                           columns=fe.feature_columns)

    fe.fit_scaler(X_train)
    scaled_valid = fe.transform(X_valid)

    # Modificar validación no debe afectar el escalador
    X_valid_modified = X_valid.copy()
    X_valid_modified.iloc[0, 0] += 999

    scaled_valid_modified = fe.transform(X_valid_modified)

    # El escalador no cambia porque no se re-ajusta
    assert np.allclose(scaled_valid.iloc[:, 1:], scaled_valid_modified.iloc[:, 1:])
