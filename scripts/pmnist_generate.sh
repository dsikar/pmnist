#!/bin/bash
#SBATCH -D /users/aczd097/git/pmnist/scripts    # Working directory
#SBATCH --job-name pmnist                      # Job name
#SBATCH --partition=gengpu                       # Select the correct partition.
#SBATCH --nodes=1                                  # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=2                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=4GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=23:59:59                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:2                               # Use two gpus.
#SBATCH -e results/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o results/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.sikar@city.ac.uk         # Where to send mail

#Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

#Remove any unwanted modules
module purge

#Modules required
#module load python/3.7.12 # now loading through pipenv
module add gnu
#Run script
start=$(date +%s) # Record the start time in seconds since epoch

#python mnist_cnn_train.py #--save-model todo
python generate_pmnist_dataset.py


end=$(date +%s) # Record the end time in seconds since epoch
diff=$((end-start)) 

# Convert seconds to hours, minutes, and seconds
hours=$((diff / 3600))
minutes=$(( (diff % 3600) / 60 ))
seconds=$((diff % 60))

echo "python generate_pmnist_dataset.py - Script execution time: $hours hours, $minutes minutes, $seconds seconds"

