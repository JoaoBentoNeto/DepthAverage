import numpy as np
import matplotlib.pyplot as plt
import analitico_2d_depth
import analitico_2d_grey
import lbm_autoral
import analitico_3d
import ler_vtk

if __name__ == "__main__":
    # Parâmetros do paper
    tau = 0.9330127
    Ny = 80
    h = 8
    g_lat = [1.0e-8, 0.0]

    # Média em profundidade analítico
    u_analitico_2d_depth = (
        analitico_2d_depth.perfil_velocidade_analitico_2d_depth(
            tau, Ny, g_lat[0], h
        )
    )
    k_analitico_2d_depth, k_analitico_2d_depth_mD = (
        analitico_2d_depth.permeabilidade_analitica_2d_depth(
            tau, Ny, g_lat[0], h
        )
    )

    # Média em profundidade código autoral
    u_lbm_autoral = lbm_autoral.perfil_velocidade_lbm_autoral(
        tau, Ny, g_lat, h, 1
    )
    k_lbm_autoral, k_lbm_autoral_mD = lbm_autoral.permeabilidade_lbm_autoral(
        tau, Ny, g_lat, h, 1
    )

    # Média volumétrica analítico
    u_analitico_2d_grey = analitico_2d_grey.perfil_velocidade_analitico_2d_grey(
        tau, Ny, g_lat[0], h
    )
    k_analitico_2d_grey, k_analitico_2d_grey_mD = (
        analitico_2d_grey.permeabilidade_analitica_2d_grey(tau, Ny, g_lat[0], h)
    )

    # # Média volumétrica LBPM
    # u_lbpm_2d = ler_vtk.perfil_velocidade_lbpm(
    #     "/home/bento/remote/hal/hele-shaw/duto_quadrado/duto_grey_20"
    # )
    # k_lbpm_2d, k_lbpm_2d_mD = ler_vtk.permeabilidade_lbpm(
    #     "/home/bento/remote/hal/hele-shaw/duto_quadrado/duto_grey_20"
    # )

    # Analítico 3D
    u_analitico_3d = analitico_3d.perfil_velocidade_analitico_3d(
        tau, Ny, g_lat[0], h
    )
    k_analitico_3d, k_analitico_3d_mD = (
        analitico_3d.permeabilidade_analitica_3d(tau, Ny, g_lat[0], h)
    )

    # # LBPM 3D
    # u_lbpm_3d = ler_vtk.perfil_velocidade_lbpm(
    #     "/home/bento/remote/hal/hele-shaw/duto_quadrado/duto_full_20"
    # )
    # k_lbpm_3d, k_lbpm_3d_mD = ler_vtk.permeabilidade_lbpm(
    #     "/home/bento/remote/hal/hele-shaw/duto_quadrado/duto_full_20"
    # )

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

    a = Ny / 2
    b = h / 2
    fator_correcao = (4 * a * b) / ((2 * a + 2) * (2 * b + 2)) * (Ny + 2) / Ny
    # =============================================================================
    # 3. Geração e Impressão da Tabela de Erros
    # =============================================================================

    dados_tabela = [
        (
            "Analítico 3D",
            k_analitico_3d,
            k_analitico_3d_mD,
            u_analitico_3d,
            x_analitico,
        ),
        (
            "Analítico 2D (Profundidade)",
            k_analitico_2d_depth,
            k_analitico_2d_depth_mD,
            u_analitico_2d_depth,
            x_analitico,
        ),
        (
            "LBM Autoral 2D",
            k_lbm_autoral,
            k_lbm_autoral_mD,
            u_lbm_autoral,
            x_autoral,
        ),
        (
            "Analítico 2D (Volumétrico)",
            k_analitico_2d_grey,
            k_analitico_2d_grey_mD,
            u_analitico_2d_grey,
            x_analitico,
        ),
        # ("LBPM 2D", k_lbpm_2d, k_lbpm_2d_mD, u_lbpm_2d, x_lbpm),
        # ("LBPM 3D", k_lbpm_3d, k_lbpm_3d_mD, u_lbpm_3d, x_lbpm),
    ]

    print("-" * 120)
    print(
        f"{'Descrição (Caso)':<30} | {'Perm. (um²)':<15} | {'Perm. (mD)':<15} | {'Erro Perm (%)':<15} | {'Erro Vel (%)':<15}"
    )
    print("-" * 120)

    for desc, k_val, k_md, u_array, x_array in dados_tabela:
        if k_val is None or k_md is None:
            print(
                f"{desc:<30} | {'Falhou':<15} | {'Falhou':<15} | {'-':<15} | {'-':<15}"
            )
            continue

        # 1. Erro de Permeabilidade
        erro_k = abs(k_val - k_analitico_3d) / k_analitico_3d * 100

        # 2. Erro do Perfil de Velocidade
        u_3d_interp = np.interp(x_array, x_analitico, u_analitico_3d)
        erro_v = np.mean(abs(u_array - u_3d_interp) / u_3d_interp) * 100

        print(
            f"{desc:<30} | {k_val:<15.6e} | {k_md:<15.4f} | {erro_k:<15.4f} | {erro_v:<15.4f}"
        )

    print("-" * 120)

    # =============================================================================
    # 3. Plotagem dos Perfis de Velocidade
    # =============================================================================
    plt.figure(figsize=(10, 8))

    # --- Plot dos Analíticos (Linhas contínuas e tracejadas) ---
    plt.plot(
        x_analitico,
        u_analitico_2d_depth / np.max(u_analitico_3d),
        "-",
        linewidth=2,
        color="blue",
        label="Analítico 2D (Média em Profundidade)",
    )
    plt.plot(
        x_analitico,
        u_analitico_2d_grey / np.max(u_analitico_3d),
        "--",
        linewidth=2,
        color="green",
        label="Analítico 2D (Média Volumétrica/Grey)",
    )
    plt.plot(
        x_analitico,
        u_analitico_3d / np.max(u_analitico_3d),
        ":",
        linewidth=2,
        color="black",
        label="Analítico 3D",
    )

    # --- Plot dos Numéricos (Marcadores/Pontos) ---
    plt.plot(
        x_autoral,
        u_lbm_autoral / np.max(u_analitico_3d),
        "o",
        markersize=6,
        color="dodgerblue",
        # markerfacecolor="none",
        label="LBM Autoral (Média em Profundidade)",
    )
    # plt.plot(
    #     x_lbpm,
    #     u_lbpm_2d,
    #     "s",
    #     markersize=8,
    #     color="limegreen",
    #     markerfacecolor="none",
    #     label="LBPM 2D (Média Volumétrica/Grey)",
    # )
    # plt.plot(
    #     x_lbpm,
    #     u_lbpm_3d,
    #     "^",
    #     markersize=8,
    #     color="red",
    #     markerfacecolor="none",
    #     label="LBPM 3D",
    # )

    plt.xlabel("Posição ao longo da largura (y)")
    plt.ylabel(
        r"Velocidade normalizada ($\frac{v_z}{v_{z,max,3D\ analítico}}$)"
    )
    plt.title("Comparação dos Perfis de Velocidade")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(
        f"/home/bento/remote/hal/depth_average/all_codes/duto_quadrado/comparacao_perfis_velocidade{Ny}x{h}.png",
        dpi=300,
    )
    plt.show()
