#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=07:00:00
#SBATCH --account=def-rmelnik
#SBATCH --output=myslurm_%j.log
##
pwd; hostname; date
##
module --force purge && module load StdEnv/2016.4 && module load nixpkgs/16.09 intel/2019.3 intelmpi/2019.3.199 && module load python/3.6.3 && source /home/chinchay/projects/def-rmelnik/chinchay/mydocs/venvs/jupyter_py3/bin/activate



srun python r15.py >> outpy



