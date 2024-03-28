#!/bin/bash

scp dsbloom.* jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/

ssh jeanzay << 'ENDLULLY'
cd /gpfswork/rech/knb/uyr14tk/home/openllmfr/
sbatch dsbloom.slurm
ENDLULLY

