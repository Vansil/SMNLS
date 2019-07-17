#!/bin/bash

for filename in *.sh; do
    for ((el=2; el<=3; el++)); do
        for ((i=1; i<=6; i++)); do
            outfile=${filename::-3}_elmo"$el"_seed"$i".sh
            sed -e 's:ELMO:'$el':g' "$filename" | sed -e 's:SEED:'$i':g' > $outfile
            cat $outfile
        done
    done
done
