## Raw speed results

### On Jean Zay

- FSDP, bloom-7b, batchsize=2, context=1024, 8xA100: 200 jours pour 100b tokens
    - FSDP sur 16xA100: 2x plus lent, car interconnection entre les noeuds trop lente ! Mais on pourrait tenter avec gradient accumulation…

- Deepspeed stage 1, AdamW, bloom-7b, bs=1, context=1024, 8xA100: 258 jours pour 100b tokens
- Deepspeed stage 1, AdamW, bloom-7b, bs=2, context=1024, 16xA100:  2x plus lent!

- DDP+gradient checkpointing+adafactor, bloom-7b, bs=1, context=1024, 2xA100: 1200 jours pour 100b tokens
- DDP+gradient checkpointing+adafactor, bloom-7b, bs=1, context=1024, 8xA100: 350 jours pour 100b tokens
    - gradient checkpointing + diff(DDP,DS1) coute +36% en temps de calcul
- DDP+gradient checkpointing+adafactor, bloom-7b, bs=1, context=1024, 16xA100: 700 jours pour 100b tokens
    - Toujours ce facteur X2 sur 2 nodes alors que c’est du DDP de base !
- DDP+gradient checkpointing+adafactor+gradient compression (powerSGD), bloom-7b, bs=1, context=1024, 16xA100: 500 jours pour 100b tokens
- DDP+gradient checkpointing+adafactor+gradient compression (powerSGD), bloom-7b, bs=2, context=1024, 8xA100: 335 jours pour 100b tokens
- DDP+gradient checkpointing+adafactor+gradient compression (powerSGD) + gradient accumulation (256), bloom-7b, bs=2, context=1024, 16xA100:  73 jours pour 100b tokens
- DDP+gradient checkpointing+adafactor+gradient compression (powerSGD) + gradient accumulation (256), bloom-7b, bs=2, context=1024, 32xA100: 36 jours pour 100b tokens
- DDP+gradient checkpointing+adafactor+gradient compression (powerSGD) + gradient accumulation (256), bloom-7b, bs=2, context=2048, 32xA100: 40 jours pour 100b tokens

- Deepspeed stage 1+ gradacc 256, AdamW, bloom-7b, bs=2, context=1024, 16xA100:  57 jours pour 100b tokens
- Deepspeed stage 1+ gradacc 256, AdamW, bloom-7b, bs=2, context=1024, 32xA100:  **29 jours pour 100b tokens**
- Deepspeed stage 1+ gradacc 256, AdamW, bloom-7b, bs=2, context=1024, 40xA100:  **23 jours pour 100b tokens**
- Deepspeed stage 1+ gradacc 256, AdamW, bloom-7b, bs=2, context=1024, 48xA100:  **19 jours pour 100b tokens**
- Deepspeed stage 2+ gradacc 256, AdamW, bloom-7b, bs=2, context=2048, 32xA100: crash OOM, de plus, 20% des GPU sont à 0% alors que dans les tests precedents, tous les GPU sont a 100%
- Deepspeed stage 1+ gradacc 256 + gradient checkpointing, AdamW, bloom-7b, bs=2, context=2048, 32xA100: **40 jours pour 100b tokens**

### On Adastra

- FSDP, bloom-7b, batchsize=2, context=1024, 2xMI150X: 1500 jours pour 100b tokens
- FSDP, bloom-7b, batchsize=2, context=1024, 4xMI150X: crash OOM

## Preliminary loss curves

- FSDP: bonne convergence

![fsdplog.png](imgs/fsdplog.png)

- Deepspeed: idem
- DDP avec Adafactor: il train, mais plus difficile a tuner !

![ddp.png](imgs/ddp.png)

- DDP + grad accu 256: needs some tuning for sure, but this curve shows clearly the compromise to find between compensating for inter-node slowness and trying to not impact convergence speed too much…

![gradacc.png](imgs/gradacc.png)

Deepspeed + gradaccu 256 (pas de gradient checkpointing !): meilleure convergence grace a Adam, et devrait etre plus rapide a GPU egaux grace a gradient accumuluation: preferred option

![ds.png](imgs/ds.png)
