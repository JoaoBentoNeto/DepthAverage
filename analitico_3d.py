import numpy as np
import matplotlib.pyplot as plt


def permeabilidade_analitica_3d(tau, Ny, g_lat, h):
    """
    Calcula a permeabilidade analítica para um duto 3D (Full).
    """
    a = Ny / 2
    b = h / 2
    S = 0.0
    n_terms = 100  # Número de termos na série
    for n in range(1, n_terms * 2, 2):
        S += np.tanh(n * np.pi * b / (2 * a)) / n**5
    k = a**2 / 3 - (64 * a**3) / (b * np.pi**5) * S

    # u_bar = 0.0
    # a = Ny / 2
    # b = h / 2
    # S = 0.0
    # nu = (tau - 0.5) / 3.0
    # coef = 16 * g_lat * a**2 / (nu * np.pi**3)
    # n_terms = 100

    # numero_pontos = 3000  # Resolução aumentada para precisão na média
    # x = np.linspace(0, Ny, numero_pontos)

    # # --- CORREÇÃO AQUI ---
    # # Deslocamos o 'x' para que o centro do duto seja matematicamente o 0
    # x_math = x - a

    # for n in range(1, n_terms * 2, 2):  # Apenas ímpares
    #     # Termo resultante da integração na profundidade
    #     termo_int = 1.0 - (2 * a / (n * np.pi * b)) * np.tanh(
    #         n * np.pi * b / (2 * a)
    #     )

    #     termo_n = (
    #         ((-1) ** ((n - 1) // 2) / n**3)
    #         * termo_int
    #         * np.cos(n * np.pi * x_math / (2 * a))  # Usa o x_math aqui!
    #     )
    #     u_bar += termo_n

    # k = np.mean(u_bar * coef) * nu / g_lat

    # Correção para o LBM: considerando os voxeis sólidos de borda
    fator_correcao = (4 * a * b) / ((2 * a + 2) * (2 * b + 2))
    k_final = k #* fator_correcao

    return k_final, k_final / 0.0009869233


def perfil_velocidade_analitico_3d(tau, Ny, g_lat, h):
    """
    Calcula a velocidade média na profundidade u_bar(x)
    para a seção transversal de um duto retangular (integração em y).
    """
    u_bar = 0.0
    a = Ny / 2
    b = h / 2
    S = 0.0
    nu = (tau - 0.5) / 3.0
    coef = 16 * g_lat * a**2 / (nu * np.pi**3)
    n_terms = 100

    numero_pontos = 3000  # Resolução aumentada para precisão na média
    x = np.linspace(0, Ny, numero_pontos)

    # --- CORREÇÃO AQUI ---
    # Deslocamos o 'x' para que o centro do duto seja matematicamente o 0
    x_math = x - a

    for n in range(1, n_terms * 2, 2):  # Apenas ímpares
        # Termo resultante da integração na profundidade
        termo_int = 1.0 - (2 * a / (n * np.pi * b)) * np.tanh(
            n * np.pi * b / (2 * a)
        )

        termo_n = (
            ((-1) ** ((n - 1) // 2) / n**3)
            * termo_int
            * np.cos(n * np.pi * x_math / (2 * a))  # Usa o x_math aqui!
        )
        u_bar += termo_n

    return u_bar * coef


if __name__ == "__main__":
    # parâmetros do paper
    tau = 1.1
    Ny = 20
    h = 20.0
    g_lat = [1.0e-8, 0.0]

    # Analítico 3D
    u_analitico_3d = perfil_velocidade_analitico_3d(tau, Ny, g_lat[0], h)
    k_analitico_3d, k_analitico_3d_mD = permeabilidade_analitica_3d(
        tau, Ny, g_lat[0], h
    )

    # =============================================================================
    # 1. Definição dos eixos X (Coordenadas)
    # =============================================================================
    # Analíticos: 3000 pontos, de 0 a Ny
    numero_pontos = 3000
    x_analitico = np.linspace(0, Ny, numero_pontos)

    # LBPM: Ny + 2 pontos, indo da borda exterior de bounce-back (-0.5 a Ny+0.5)
    x_lbpm = np.linspace(-0.5, Ny + 0.5, Ny + 2)

    # LBM Autoral: Ny pontos, no centro das células (0.5 a Ny-0.5)
    x_autoral = np.linspace(0.5, Ny - 0.5, Ny)

    # =============================================================================
    # 2. Plotagem dos Perfis de Velocidade
    # =============================================================================
    plt.figure(figsize=(10, 8))

    # --- Plot dos Analíticos (Linhas contínuas e tracejadas) ---

    plt.plot(
        x_analitico,
        u_analitico_3d,
        ":",
        linewidth=2,
        color="black",
        label="Analítico 3D",
    )

    # Configurações estéticas do gráfico
    plt.xlabel("Posição ao longo da largura (y)")
    plt.ylabel("Velocidade média ($v_z$)")
    plt.title("Comparação dos Perfis de Velocidade")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(
        "comparacao_perfis_velocidade.png", dpi=300
    )  # Salva o gráfico como imagem de alta resolução
    plt.show()
