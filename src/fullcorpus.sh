#!/bin/bash

scp fullcorpus.* jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/

ssh jeanzay << 'ENDLULLY'
cd /gpfswork/rech/knb/uyr14tk/home/openllmfr/
rm fnoms
find /gpfswork/rech/qgz/commun/data/corpus_openllm/ -name "*.parquet" | grep -v -e perplexity_corpus_open_llm -e 'croissant_aligned/train/data-00039-of-00080' > fnoms
sbatch fullcorpus.slurm
ENDLULLY

