#!/bin/bash
#SBATCH --job-name=lsss
#SBATCH --output=/home/yifanjin/lsss1/lsss-%j.out
#SBATCH --error=/home/yifanjin/lsss1/lsss-%j.err
#SBATCH --mail-user=yifanjin@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --account=stats_dept1
#SBATCH --mem-per-cpu=2G      # increase as needed
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --array=0-999

RUNPATH=/home/yifanjin/ENV/lib/python3.9/site-packages/lsss-master/lsss/
cd $RUNPATH

module load python/3.9.7
# module load git/2.20.1
# module load gcc/9.2.0
source $HOME/ENV/bin/activate
# pip install --no-index --upgrade pip
# python -m pip install numpy
# python -m pip install scipy
# python -m pip install typing
# python -m pip install dataclasses
# python -m pip install msprime
# python -m pip install joblib

# git clone https://github.com/vcftools/vcftools.git
# cd vcftools
# ./autogen.sh
# ./configure prefix=$HOME
# make
# make install


# git clone --recursive https://github.com/brentp/cyvcf2
# cd cyvcf2/htslib
# autoheader
# autoconf
# ./configure --enable-libcurl
# make
# cd ..
# pip install -r requirements.txt
# CYTHONIZE=1 pip install -e .
# cd ..

# cd $RUNPATH
#can be used for any experiment, only need to change the name
python prephased_loss2.py $SLURM_ARRAY_TASK_ID
