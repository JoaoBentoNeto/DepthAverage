import numpy as np
import matplotlib.pyplot as plt
import re
import os
import glob
import pyvista as pv

# =============================================================================
# Configurações de Plotagem
# =============================================================================
TAMANHO_BASE = 18
plt.rcParams.update(
    {
        "font.size": TAMANHO_BASE,
        "axes.titlesize": TAMANHO_BASE + 4,
        "axes.labelsize": TAMANHO_BASE + 2,
        "xtick.labelsize": TAMANHO_BASE,
        "ytick.labelsize": TAMANHO_BASE,
        "legend.fontsize": TAMANHO_BASE,
        "figure.titlesize": TAMANHO_BASE + 6,
    }
)

# Patch para compatibilidade de versões mais recentes do NumPy
if not hasattr(np, "bool"):
    np.bool = bool


# =============================================================================
# Extração de Dados do Banco de Dados (.db)
# =============================================================================
def extract_values_from_db(file_path):
    """Extrai valores do arquivo de configuração .db do LBPM."""
    values = {}
    try:
        with open(file_path, "r") as file:
            content = file.read()

        patterns = {
            "tau": r"tau\s*=\s*([\d.]+)",
            "F": r"F\s*=\s*([^,]+),\s*([^,]+),\s*([^\s,]+)",
            "N": r"N\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
            "porosity": r"PorosityList\s*=\s*([\d.,\s]+)",
            "permeability": r"PermeabilityList\s*=\s*([\d.,\s]+)",
            "h": r"voxel_length\s*=\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                if key == "tau":
                    values["tau"] = float(match.group(1))
                elif key == "F":
                    values["Fx"] = float(match.group(1))
                    values["Fy"] = float(match.group(2))
                    values["Fz"] = float(match.group(3))
                elif key == "N":
                    values["Nx"] = int(match.group(1))
                    values["Ny"] = int(match.group(2))
                    values["Nz"] = int(match.group(3))
                elif key == "porosity":
                    values["porosity"] = [
                        float(x.strip()) for x in match.group(1).split(",")
                    ]
                elif key == "permeability":
                    values["permeability"] = [
                        float(x.strip()) for x in match.group(1).split(",")
                    ]
                elif key == "h":
                    values["h"] = float(match.group(1))

        return values

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {file_path}")
        return {}
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")
        return {}


# =============================================================================
# Funções Principais Solicitadas
# =============================================================================


def permeabilidade_lbpm(pasta_simulacao):
    """
    Procura o arquivo output_XX.log com o maior XX dentro da pasta da simulação.
    Extrai o último valor de permeabilidade (em mD) registrado e converte para Darcy.

    Retorna: (permeabilidade_darcy, permeabilidade_mD)
    """
    if not os.path.exists(pasta_simulacao):
        print(f"Erro: Pasta '{pasta_simulacao}' não encontrada.")
        return None, None

    arquivos = []
    for f in os.listdir(pasta_simulacao):
        match = re.match(r"output_(\d+)\.log", f)
        if match:
            arquivos.append((int(match.group(1)), f))

    if not arquivos:
        print("Erro: Nenhum arquivo output_XX.log encontrado.")
        return None, None

    # Pega o arquivo com maior número XX
    maior_arquivo = sorted(arquivos, key=lambda x: x[0])[-1][1]
    caminho_arquivo = os.path.join(pasta_simulacao, maior_arquivo)

    ultimo_valor_md = None
    linha_anterior = ""
    regex_tracos = re.compile(r"^-+$")
    regex_numero = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as file:
            for linha in file:
                linha_limpa = linha.strip()

                if regex_tracos.fullmatch(linha_limpa) and linha_anterior:
                    if "[mD]" in linha_anterior:
                        parte_antes = linha_anterior.split("[mD]")[0]
                        numeros = regex_numero.findall(parte_antes)
                        if numeros:
                            ultimo_valor_md = float(numeros[-1])
                    else:
                        numeros = regex_numero.findall(linha_anterior)
                        if numeros:
                            ultimo_valor_md = float(numeros[-1])

                if linha_limpa:
                    linha_anterior = linha_limpa

    except Exception as e:
        print(f"Erro ao ler o log: {e}")
        return None, None

    if ultimo_valor_md is not None:
        # Fator de conversão aproximado de mD para um2
        valor_um2 = ultimo_valor_md * 0.0009869233
        return valor_um2, ultimo_valor_md
    else:
        print("Valor de permeabilidade não encontrado no log.")
        return None, None


def perfil_velocidade_lbpm(pasta_simulacao):
    """
    Encontra a pasta vis* mais recente na simulação, lê o arquivo summary.pvti
    e retorna o perfil de velocidade em `x` (largura) tirando a média
    em `y` (profundidade) e em `z` (direção do fluxo).

    Retorna: Um array 1D com o perfil de velocidades ao longo da largura (x).
    """
    # Encontrar pastas 'vis*' dentro da pasta da simulação
    padrao_vis = os.path.join(pasta_simulacao, "vis*")
    vis_folders = glob.glob(padrao_vis)

    if not vis_folders:
        raise FileNotFoundError(
            f"Nenhuma pasta vis* encontrada em {pasta_simulacao}."
        )

    # Ordenar numericamente para pegar a mais recente
    def extract_number(folder_name):
        nums = re.findall(r"\d+", os.path.basename(folder_name))
        return int(nums[0]) if nums else -1

    vis_folders.sort(key=extract_number)
    latest_vis = vis_folders[-1]

    grid_file = os.path.join(latest_vis, "summary.pvti")
    if not os.path.exists(grid_file):
        raise FileNotFoundError(f"Arquivo {grid_file} não encontrado.")

    print(f"Lendo dados espaciais de: {grid_file}")
    grid = pv.read(grid_file)

    # Extração e redimensionamento da malha de células
    nx, ny, nz = grid.dimensions
    shape = (nx - 1, ny - 1, nz - 1)

    # Extrair velocidade na direção z (fluxo)
    vz_flat = grid.cell_data["Velocity_z"]
    vz_3d = vz_flat.reshape(shape, order="F")

    # Mapeamento do shape em python:
    # Eixo 0 = x (largura)
    # Eixo 1 = y (profundidade)
    # Eixo 2 = z (fluxo)
    #
    # Para ter um perfil em "x" onde calculamos a média de y e z:
    perfil_x = np.mean(vz_3d, axis=(1, 2))

    return perfil_x
