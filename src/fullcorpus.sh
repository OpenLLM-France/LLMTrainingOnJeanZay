#!/bin/bash

. ~/envs/transformers/bin/activate

for f in `cat fnoms`; do
    echo $f
    a=`basename $f`
    echo $a
    # scp jeanzay:$f ./
    python fullcorpus.py $a 
    scp $a".txt" jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/alldata/
    exit
    rm $a
    rm $a".txt"
done

