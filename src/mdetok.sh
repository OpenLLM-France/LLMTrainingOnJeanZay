#!/bin/bash

scp mdetok.* jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/

ssh jeanzay << 'ENDLULLY'
cd /gpfswork/rech/knb/uyr14tk/home/openllmfr/
sbatch mdetok.slurm
ENDLULLY

