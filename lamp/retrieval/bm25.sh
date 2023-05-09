dataset_dir=$1 #data/translation/jrc_acquis/ende
output_dir=$2 #indexes/jrc_acquis
query=$3 #en
lang=$query

python lamo/retrieval/build_pyserini_input.py \
    --dataset_dir ${dataset_dir} \
    --query en \
    --output_dir ${output_dir}

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${output_dir}/document  \
  --index ${output_dir}/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 10 \
  --language $lang \
  --storePositions --storeDocvectors --storeRaw 

for _split in train dev test
do
  python -m pyserini.search.lucene \
    --index ${output_dir}/index \
    --topics ${output_dir}/query/${_split}_query.tsv \
    --output ${output_dir}/${_split}_results.trec \
    --bm25 \
    --language $lang \
    --thread 10 \
    --batch-size 160 \
    --hits 5
done

#rm -rf ${output_dir}/document ${output_dir}/query ${output_dir}/index
