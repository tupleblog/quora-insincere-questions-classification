for i in `seq 0 4`; do
    allennlp train experiments/quora_$i.json -s output_$i --include-package quora
done