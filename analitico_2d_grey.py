import numpy as np


def perfil_velocidade_analitico_2d_grey(tau, Ny, g_lat, h):
    """
    Calcula o perfil de velocidade analítica para o duto quadrado pela
    aproximação em média volumetrica (Grey-Poiseuille).
    """

    nu = (tau - 0.5) / 3.0

    k = (h**2) / 12.0
    phi = 0.9999

    # Lambda e termo particular
    lambda_ = np.sqrt(phi / k) if k > 0 else 0
    U_particular = (k / nu) * g_lat

    # Montagem do Sistema Linear (2x2) para u(0) = 0 e u(Ny) = 0
    M = np.array([[1.0, 1.0], [np.exp(lambda_ * Ny), np.exp(-lambda_ * Ny)]])
    b_vec = np.array([-U_particular, -U_particular])

    # Resolução dos coeficientes A e B
    A, B = np.linalg.solve(M, b_vec)

    # Construção do array de velocidades
    numero_pontos = 3000  # Resolução aumentada para precisão na média
    y = np.linspace(0, Ny, numero_pontos)
    u = A * np.exp(lambda_ * y) + B * np.exp(-lambda_ * y) + U_particular

    return u


def permeabilidade_analitica_2d_grey(tau, Ny, g_lat, h_aperture_lat):

    u = perfil_velocidade_analitico_2d_grey(tau, Ny, g_lat, h_aperture_lat)
    nu = (tau - 0.5) / 3.0

    # Permeabilidade média e aplicação da correção LBM
    k_anal = (np.mean(u)) * nu / g_lat
    fator_correcao = Ny / (Ny + 2.0)
    k_final = k_anal  # * fator_correcao

    return k_final, k_final / 0.0009869233
