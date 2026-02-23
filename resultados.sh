#!/bin/bash
#SBATCH --job-name=duto_quadrado
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=close_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:00:00

# Carregar o módulo do Conda
module load conda

# Ativar o ambiente Conda
conda activate poropy

# Executar o script Python
python -u resultados.py
