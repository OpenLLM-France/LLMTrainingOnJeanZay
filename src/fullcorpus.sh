#!/bin/bash

. ~/envs/transformers/bin/activate

for f in `cat fnoms`; do
    echo $f
    c=`echo $f | cut -c1`
    if [ $c == '#' ]; then
        echo "pass file"
    else
        a=`basename $f`
        echo $a
        scp jeanzay:$f ./
        python fullcorpus.py $a 
        gzip $a".txt"
        scp $a".txt.gz" jeanzay:/gpfswork/rech/knb/uyr14tk/home/openllmfr/alldata/
        rm $a"*"
    fi
done

