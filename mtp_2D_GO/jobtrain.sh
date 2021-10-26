#!/bin/sh
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:30:00
#SBATCH --account=def-rmelnik
#SBATCH --output=myslurm_%j.log
##
pwd; hostname; date
##
module --force purge && module load StdEnv/2016.4 && module load nixpkgs/16.09 intel/2019.3 intelmpi/2019.3.199 && module load python/3.6.3 && source /home/chinchay/projects/def-rmelnik/chinchay/mydocs/venvs/jupyter_py3/bin/activate


echo "_ jobtrain.sh: starting training at date:" >> ../mylog.txt
date >> ../mylog.txt

srun mlp train pot.mtp train.cfg > training.txt

echo "_ finished training at date:" >> mylog.txt
date >> ../mylog.txt
echo "_ ..." >> ../mylog.txt

