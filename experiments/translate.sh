spm_encode --model=../data/transformer/sentencepiece.model < $1 > $1.atok
python ../translate.py -model ../data/transformer/averaged-10-epoch.pt -src $1.atok -output $1_output.atok -verbose
rm *.atok
