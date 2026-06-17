import numpy as np
import gudhi as gd


def build_periodic_complex(curve):
    """
    Construye un complejo simplicial equivalente a un ciclo S¹.

    Cada punto de la curva es un vértice.
    Se agregan aristas consecutivas y una arista final
    que conecta el último vértice con el primero.
    """

    curve = np.asarray(curve, dtype=float)

    st = gd.SimplexTree()

    n = len(curve)

    # Vértices
    for i in range(n):
        st.insert([i], filtration=float(curve[i]))

    # Aristas consecutivas
    for i in range(n):
        j = (i + 1) % n

        filtration_value = max(
            float(curve[i]),
            float(curve[j])
        )

        st.insert(
            [i, j],
            filtration=filtration_value
        )

    st.make_filtration_non_decreasing()

    return st


def persistence_diagram(curve):
    """
    Calcula persistencia H0 y H1
    para una curva de Andrews discretizada.
    """

    st = build_periodic_complex(curve)

    persistence = st.persistence()

    h0 = st.persistence_intervals_in_dimension(0)
    h1 = st.persistence_intervals_in_dimension(1)

    return h0, h1


def persistence_statistics(intervals):
    """
    Calcula estadísticas básicas
    sobre un conjunto de intervalos.
    """

    if len(intervals) == 0:
        return {
            "num_pairs": 0,
            "total_persistence": 0.0,
            "max_life": 0.0,
            "entropy": 0.0
        }

    lifetimes = []

    for birth, death in intervals:

        if np.isinf(death):
            continue

        life = death - birth

        if life > 0:
            lifetimes.append(life)

    lifetimes = np.array(lifetimes)

    if len(lifetimes) == 0:
        return {
            "num_pairs": 0,
            "total_persistence": 0.0,
            "max_life": 0.0,
            "entropy": 0.0
        }

    total_persistence = np.sum(lifetimes)

    probs = lifetimes / total_persistence

    entropy = -np.sum(
        probs * np.log(probs)
    )

    return {
        "num_pairs": len(lifetimes),
        "total_persistence": float(total_persistence),
        "max_life": float(np.max(lifetimes)),
        "entropy": float(entropy)
    }