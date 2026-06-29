"""Persistencia de subniveles para curvas de Andrews con periodicidad S¹.

Construye un complejo simplicial periódico a partir de una curva
discretizada y extrae diagramas de persistencia H₀ y H₁ usando GUDHI.

Funciones
---------
build_periodic_complex  — complejo simplicial periódico S¹
persistence_diagram     — diagramas H₀ y H₁ para una curva
persistence_statistics  — estadísticos sobre intervalos (num_pairs, total, max, entropy)
h1_global_interval      — intervalo global H₁ (min, max)
"""

import numpy as np
import gudhi as gd


def build_periodic_complex(curve):
    """Complejo simplicial periódico (ciclo S¹) a partir de una curva.

    Cada punto de la curva es un vértice. Se agregan aristas entre
    puntos consecutivos y una arista final que cierra el ciclo.

    Parameters
    ----------
    curve : array-like, shape (N,)
        Curva discretizada en N puntos.

    Returns
    -------
    st : gudhi.SimplexTree
    """
    curve = np.asarray(curve, dtype=float)
    n = len(curve)

    st = gd.SimplexTree()

    for i in range(n):
        st.insert([i], filtration=float(curve[i]))

    for i in range(n):
        j = (i + 1) % n
        st.insert([i, j], filtration=max(float(curve[i]), float(curve[j])))

    st.make_filtration_non_decreasing()
    return st


def persistence_diagram(curve):
    """Diagramas de persistencia H₀ y H₁ para una curva de Andrews.

    Parameters
    ----------
    curve : array-like
        Curva discretizada en N puntos. Si se pasa un escalar,
        se envuelve automáticamente en un array 1D para evitar errores.

    Returns
    -------
    h0 : ndarray, shape (k, 2)
        Intervalos de persistencia H₀ (birth, death).
    h1_gudhi : ndarray
        Intervalos H₁ detectados por GUDHI (usualmente vacío en 1D).
    h1_global : ndarray, shape (1, 2)
        Intervalo global H₁ = [min(curve), max(curve)].
    """
    curve = np.asarray(curve, dtype=float)
    if curve.ndim == 0:
        curve = curve.reshape(1)

    st = build_periodic_complex(curve)
    st.persistence()

    h0 = st.persistence_intervals_in_dimension(0)
    h1_gudhi = st.persistence_intervals_in_dimension(1)

    h1_global = np.array([[float(np.min(curve)), float(np.max(curve))]])

    return h0, h1_gudhi, h1_global


def persistence_statistics(intervals):
    """Estadísticos básicos sobre un conjunto de intervalos de persistencia.

    Parameters
    ----------
    intervals : ndarray, shape (m, 2)
        Intervalos (birth, death). Se ignoran los infinitos.

    Returns
    -------
    dict con llaves: num_pairs, total_persistence, max_life, entropy
    """
    if len(intervals) == 0:
        return {"num_pairs": 0, "total_persistence": 0.0,
                "max_life": 0.0, "entropy": 0.0}

    lifetimes = []
    for birth, death in intervals:
        if np.isinf(death):
            continue
        life = death - birth
        if life > 0:
            lifetimes.append(life)

    if len(lifetimes) == 0:
        return {"num_pairs": 0, "total_persistence": 0.0,
                "max_life": 0.0, "entropy": 0.0}

    lifetimes = np.array(lifetimes)
    total = np.sum(lifetimes)
    probs = lifetimes / total

    return {
        "num_pairs": len(lifetimes),
        "total_persistence": float(total),
        "max_life": float(np.max(lifetimes)),
        "entropy": float(-np.sum(probs * np.log(probs)))
    }


def h1_global_interval(curve):
    """Clase global H₁ de la curva periódica [min, max].

    Según el enunciado del proyecto, la persistencia H₁ global es
    max(f) - min(f).

    Parameters
    ----------
    curve : array-like, shape (N,)

    Returns
    -------
    dict con llaves: birth, death, lifetime, interval
    """
    curve = np.asarray(curve, dtype=float)
    birth = float(np.min(curve))
    death = float(np.max(curve))

    return {
        "birth": birth,
        "death": death,
        "lifetime": death - birth,
        "interval": np.array([[birth, death]])
    }
