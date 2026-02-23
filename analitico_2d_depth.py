import numpy as np
import matplotlib.pyplot as plt


def perfil_velocidade_analitico_2d_depth(tau, Ny, g_lat, h):
    """
    Calcula o perfil de velocidade analítica para o duto quadrado pela
    aproximação em média em profundidade.
    """
    nu = (tau - 0.5) / 3.0
    numero_pontos = 3000
    y = np.linspace(0, Ny, numero_pontos)

    # Centralizando o perfil no meio do canal (Ny / 2)
    # A fórmula correta para o argumento do cosh é (posição_centralizada) * sqrt(12) / h
    termo1 = np.cosh((y - Ny / 2.0) * np.sqrt(12) / h)
    termo2 = np.cosh((Ny / 2.0) * np.sqrt(12) / h)

    u = (g_lat * h**2) / (12 * nu) * (1 - termo1 / termo2)

    return u


def permeabilidade_analitica_2d_depth(tau, Ny, g_lat, h):
    termo = h / (Ny * np.sqrt(3)) * np.tanh(Ny * np.sqrt(3) / h)
    k_final = h**2 / 12 * (1 - termo)  # * Ny / (Ny + 2.0)

    return k_final, k_final / 0.0009869233


if __name__ == "__main__":
    tau = 1.1
    Ny = 20
    h = 20.0
    g_lat = [1.0e-8, 0.0]

    u_analitico_2d_depth = perfil_velocidade_analitico_2d_depth(
        tau, Ny, g_lat[0], h
    )
    k_analitico_2d_depth, k_analitico_2d_depth_mD = (
        permeabilidade_analitica_2d_depth(tau, Ny, g_lat[0], h)
    )

    # =============================================================================
    # 2. Plotagem dos Perfis de Velocidade
    # =============================================================================

    numero_pontos = 3000
    x_analitico = np.linspace(0, Ny, numero_pontos)

    plt.figure(figsize=(10, 8))

    # --- Plot dos Analíticos (Linhas contínuas e tracejadas) ---
    plt.plot(
        x_analitico,
        u_analitico_2d_depth,
        "-",
        linewidth=2,
        color="blue",
        label="Analítico 2D (Média em Profundidade)",
    )

    # Configurações estéticas do gráfico
    plt.xlabel("Posição ao longo da largura (y)")
    plt.ylabel("Velocidade média ($v_z$)")
    plt.title("Comparação dos Perfis de Velocidade")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
