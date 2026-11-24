#!/bin/bash
# Submit the repair for several seeds:  ./submit_repair.sh 2 3 4 5 6 7 8 9 10
cd "$(dirname "$0")"
for s in "$@"; do
    echo "seed $s -> $(qsub -v SEED=$s pbs_repair.sh)"
done
