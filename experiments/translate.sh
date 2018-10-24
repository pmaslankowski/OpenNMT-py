../tools/tokenizer.perl -a -no-escape -l en -q  < $1 > $1.atok
python ../translate.py -model ../data/transformer/averaged-10-epoch.pt -src $1.atok -output $1_output.atok -verbose
