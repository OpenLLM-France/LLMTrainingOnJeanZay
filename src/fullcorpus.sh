#!/bin/bash

scp fullcorpus.* jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/
scp fnoms jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/

ssh jeanzay << 'ENDLULLY'
cd /gpfswork/rech/knb/uyr14tk/home/openllmfr/
sbatch fullcorpus.slurm
ENDLULLY

