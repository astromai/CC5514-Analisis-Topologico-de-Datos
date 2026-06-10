import numpy as np
import pandas as pd

def normalize_data(X, method="zscore"):
    """
    Normaliza los datos antes de construir las Curvas de Andrews.
    method puede ser:
    - "none": no normaliza
    - "minmax": deja cada variable entre 0 y 1
    - "zscore": resta la media y divide por desviación estándar
    - "robust": resta la mediana y divide por IQR
    """

    X = np.asarray(X, dtype=float)

    if method is None or method == "none":
        return X.copy()

    if method == "minmax":
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)

        denom = max_vals - min_vals
        denom[denom == 0] = 1.0

        return (X - min_vals) / denom

    if method == "zscore":
        mean_vals = np.mean(X, axis=0)
        std_vals = np.std(X, axis=0)

        std_vals[std_vals == 0] = 1.0

        return (X - mean_vals) / std_vals

    if method == "robust":
        median_vals = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)

        iqr = q3 - q1
        iqr[iqr == 0] = 1.0

        return (X - median_vals) / iqr

    raise ValueError("method debe ser: 'none', 'minmax', 'zscore' o 'robust'")


def andrews_curve(x, t):
    """
    Se construye la Curva de Andrews para una observación x.

    f_x(t) = x1/sqrt(2) + x2 sin(t) + x3 cos(t)
             + x4 sin(2t) + x5 cos(2t) + ...
    """
    x = np.asarray(x, dtype=float)

    curve = np.ones_like(t) * (x[0] / np.sqrt(2))

    for j in range(1, len(x)):
        freq = (j + 1) // 2

        if j % 2 == 1:
            curve += x[j] * np.sin(freq * t)
        else:
            curve += x[j] * np.cos(freq * t)

    return curve


def encode(X, N=200, normalize="zscore", order=None, return_t=False):
    """
    Convierte una matriz de datos en Curvas de Andrews.

    """
    # Si X viene como DataFrame de pandas, lo convertimos a matriz numpy.
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = np.asarray(X, dtype=float)

    # Si se entrega un orden de variables, reordenamos las columnas.
    if order is not None:
        X_values = X_values[:, order]

    # Normalizamos los datos antes de construir las curvas.
    X_norm = normalize_data(X_values, method=normalize)

    # Creamos la grilla de valores t en [-pi, pi].
    # endpoint=False evita duplicar -pi y pi, porque la curva es periódica.
    t = np.linspace(-np.pi, np.pi, N, endpoint=False)

    # Para cada fila de X_norm, construimos una Curva de Andrews.
    curves = np.array([
        andrews_curve(row, t)
        for row in X_norm
    ])

    # Si el usuario quiere también la grilla t, la devolvemos.
    if return_t:
        return curves, t

    return curves