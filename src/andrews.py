"""Curvas de Andrews para visualización y análisis topológico de datos multivariados.

Funciones
---------
normalize_data : normalización de datos (none, minmax, zscore, robust)
andrews_curve  : curva de Andrews para una observación
encode         : codifica un dataset completo en curvas de Andrews
"""

import numpy as np
import pandas as pd


def normalize_data(X, method="zscore"):
    """Normaliza las variables de un dataset.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    method : str
        "none"   — sin normalización
        "minmax" — cada variable en [0, 1]
        "zscore" — media 0, desviación 1
        "robust" — mediana 0, IQR 1

    Returns
    -------
    X_norm : ndarray, same shape as X
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
    """Evalúa la curva de Andrews para una observación x en la grilla t.

    f_x(t) = x₁/√2 + x₂ sin(t) + x₃ cos(t) + x₄ sin(2t) + x₅ cos(2t) + …

    Parameters
    ----------
    x : array-like, shape (n_features,)
    t : array-like, shape (N,)

    Returns
    -------
    curve : ndarray, shape (N,)
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
    """Convierte un dataset en curvas de Andrews.

    Parameters
    ----------
    X : DataFrame o ndarray, shape (n_samples, n_features)
    N : int
        Número de puntos de discretización en [-π, π].
    normalize : str
        Método de normalización (ver normalize_data).
    order : list | None
        Reordenamiento de columnas (índices).
    return_t : bool
        Si True, también retorna la grilla t.

    Returns
    -------
    curves : ndarray, shape (n_samples, N)
    t : ndarray, shape (N,) — solo si return_t=True
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = np.asarray(X, dtype=float)

    if order is not None:
        X_values = X_values[:, order]

    X_norm = normalize_data(X_values, method=normalize)
    t = np.linspace(-np.pi, np.pi, N, endpoint=False)

    curves = np.array([andrews_curve(row, t) for row in X_norm])

    if return_t:
        return curves, t
    return curves
