#!/bin/bash

scp fullcorpus2.* jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/

ssh jeanzay << 'ENDLULLY'
cd /gpfswork/rech/knb/uyr14tk/home/openllmfr/
sbatch fullcorpus2.slurm
ENDLULLY

