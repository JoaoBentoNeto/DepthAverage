import numpy as np
import matplotlib.pyplot as plt


def oposto(k):
    """Retorna o índice da direção oposta"""
    opostos = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}
    return opostos[k]


def halfway(tau, nx, ny, G, depth_value):

    dt = 1.0
    c2 = 1.0 / 3.0
    nu = (tau - 0.5) * c2 * dt

    depth_map = np.full(nx * ny, depth_value)

    e = np.array(
        [
            [0, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [-1, -1],
            [1, -1],
            [-1, 1],
        ]
    )
    w = np.array(
        [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )

    rho = np.ones(nx * ny)
    u = np.zeros(nx * ny)
    v = np.zeros(nx * ny)
    uh = np.zeros(nx * ny)
    vh = np.zeros(nx * ny)
    f_old = np.zeros(9 * nx * ny)

    # inicialização
    for i in range(nx):
        for j in range(ny):
            idx_node = i + nx * j
            for k in range(9):
                idx_f = k + 9 * i + 9 * nx * j
                f_old[idx_f] = w[k] * rho[idx_node]

    erro = 10e10
    anterior = 10e10
    timestep = 0

    while erro > 1e-10:
        f_new = np.zeros(9 * nx * ny)

        for i in range(nx):
            for j in range(ny):
                idx_node = i + nx * j

                rho_temp = 0.0
                sum_f_ex = 0.0
                sum_f_ey = 0.0

                for k in range(9):
                    idx_f = k + 9 * i + 9 * nx * j
                    rho_temp += f_old[idx_f]
                    sum_f_ex += f_old[idx_f] * e[k, 0]
                    sum_f_ey += f_old[idx_f] * e[k, 1]

                depth_local = depth_map[idx_node]

                # cálculo das velocidades
                denominador = 1.0 + (6.0 * dt * nu) / (depth_local**2)
                uh_local = (
                    sum_f_ex + 0.5 * dt * depth_local * G[0]
                ) / denominador
                vh_local = (
                    sum_f_ey + 0.5 * dt * depth_local * G[1]
                ) / denominador

                uh[idx_node] = uh_local
                vh[idx_node] = vh_local

                # velocidades "macroscopicas" (?)
                u[idx_node] = uh_local / depth_local
                v[idx_node] = vh_local / depth_local
                rho[idx_node] = rho_temp

                # termo de força
                Fx = (
                    depth_local * G[0] - (12.0 * nu / depth_local**2) * uh_local
                )
                Fy = (
                    depth_local * G[1] - (12.0 * nu / depth_local**2) * vh_local
                )

                uu = uh_local**2 + vh_local**2

                # colisão com termo de força
                for k in range(9):
                    idx_f = k + 9 * i + 9 * nx * j

                    eu = e[k, 0] * uh_local + e[k, 1] * vh_local
                    f_eq = w[k] * (rho_temp + 3.0 * eu + 4.5 * eu**2 - 1.5 * uu)

                    # Guo's discrete force term Fi
                    eF = e[k, 0] * Fx + e[k, 1] * Fy
                    uF = uh_local * Fx + vh_local * Fy
                    term1 = 3.0 * (e[k, 0] * Fx + e[k, 1] * Fy - uF)
                    term2 = 9.0 * eu * eF
                    Fi = w[k] * (1.0 - 0.5 / tau) * (term1 + term2)

                    f_new[idx_f] = (
                        f_old[idx_f]
                        - (1.0 / tau) * (f_old[idx_f] - f_eq)
                        + dt * Fi
                    )

        f_intermed = f_new.copy()

        # propagação
        for i in range(nx):
            for j in range(ny):
                # paredes - halfway
                if j == 0:
                    for k in [0, 1, 2, 3, 5, 8]:
                        i_next = (i + e[k, 0]) % nx
                        j_next = (j + e[k, 1]) % ny
                        idx_orig = k + 9 * i + 9 * nx * j
                        idx_dest = k + 9 * i_next + 9 * nx * j_next
                        f_new[idx_dest] = f_new[idx_orig]
                    for k in [4, 6, 7]:
                        idx_orig = k + 9 * i + 9 * nx * j
                        idx_dest = oposto(k) + 9 * i + 9 * nx * j
                        f_new[idx_dest] = f_intermed[idx_orig]

                elif j == ny - 1:
                    for k in [0, 1, 2, 4, 6, 7]:
                        i_next = (i + e[k, 0]) % nx
                        j_next = (j + e[k, 1]) % ny
                        idx_orig = k + 9 * i + 9 * nx * j
                        idx_dest = k + 9 * i_next + 9 * nx * j_next
                        f_new[idx_dest] = f_new[idx_orig]
                    for k in [3, 5, 8]:
                        idx_orig = k + 9 * i + 9 * nx * j
                        idx_dest = oposto(k) + 9 * i + 9 * nx * j
                        f_new[idx_dest] = f_intermed[idx_orig]

                # periódico
                else:
                    for k in range(9):
                        i_next = (i + e[k, 0]) % nx
                        j_next = (j + e[k, 1]) % ny
                        idx_orig = k + 9 * i + 9 * nx * j
                        idx_dest = k + 9 * i_next + 9 * nx * j_next
                        f_new[idx_dest] = f_intermed[idx_orig]

        f_old = f_new.copy()

        # erro
        timestep += 1
        atual = np.sum((u**2 + v**2) ** 0.5)
        erro = np.abs(atual - anterior) / atual
        anterior = atual
        print(f"timestep {timestep}: erro = {erro:.1e}", end="\r")

    return rho, u, v


def perfil_velocidade_lbm_autoral(tau, Ny, g_lat, h, dx):
    Nx = 3
    rho_hw, u_hw, v_hw = halfway(tau, Nx, Ny, g_lat, h)
    u_simu = np.zeros(Ny)

    for j in range(Ny):
        idx_node = (Nx // 2) + Nx * j
        u_simu[j] = u_hw[idx_node] * dx

    return u_simu


def permeabilidade_lbm_autoral(tau, Ny, g_lat, h, dx):
    Nx = 3
    rho_hw, u_hw, v_hw = halfway(tau, Nx, Ny, g_lat, h)
    nu_lat = (tau - 0.5) / 3.0
    absperm = dx**2 * nu_lat / g_lat[0] * np.sum(u_hw) / ((Ny) * Nx)
    return absperm, absperm / 0.0009869233


if __name__ == "__main__":
    # parâmetros do paper
    tau = 1.1
    Ny = 24
    Nx = 1
    dx = 1.0
    h_aperture_lat = 24.0 / dx
    g_lat = [1.0e-8, 0.0]

    # --------------- Execução ----------------
    rho_hw, u_hw, v_hw = halfway(tau, Nx, Ny, g_lat, h_aperture_lat)
    nu_lat = (tau - 0.5) / 3.0
    absperm = dx**2 * nu_lat / g_lat[0] * np.sum(u_hw) / ((Ny + 2) * Nx)
    print(f"\nabsperm = {absperm:.6f} um² = {absperm/ 0.0009869233:.6f} mD")

    # Convert velocity back to physical units for plotting
    u_simu_physical = np.zeros(Ny)
    for j in range(Ny):
        idx_node = (Nx // 2) + Nx * j
        u_simu_physical[j] = u_hw[idx_node] * dx

    y_mid = np.linspace(dx / 2, Ny * dx - dx / 2, Ny)

    # plot
    plt.figure(figsize=(10, 6))
    plt.title("Duto Quadradro")
    plt.plot(
        y_mid,
        u_simu_physical,
        "r--s",
        label="LBM 2.5D (autoral)",
    )
    plt.ylabel("Velocidade u / u_max")
    plt.xlabel("Largura do canal y / L")
    plt.legend()
    plt.grid()
    plt.show()
