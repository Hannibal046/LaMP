import os,json,csv
from lamo.utils import get_jsonl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir")
    parser.add_argument("--query")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    
    
    train_queries = [x[args.query] for x in get_jsonl(os.path.join(args.dataset_dir,'train.jsonl'))]
    dev_queries = [x[args.query] for x in get_jsonl(os.path.join(args.dataset_dir,'dev.jsonl'))]
    test_queries = [x[args.query] for x in get_jsonl(os.path.join(args.dataset_dir,'test.jsonl'))]

    os.makedirs(os.path.join(args.output_dir,'query'),exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'document'),exist_ok=True)


    ## build file for indexing from training set
    with open(os.path.join(args.output_dir,'document','document.jsonl'),'w') as f:
        for idx,q in enumerate(train_queries):
            f.write(
                json.dumps(
                    {
                        'id':str(idx),
                        "contents":q,
                    }
                )+'\n'
            )

    ## build queries
    
    with open(os.path.join(args.output_dir,'query','train_query.tsv'),'w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows([[str(idx),x.replace("\r","")] for idx,x in enumerate(train_queries)])

    with open(os.path.join(args.output_dir,'query','dev_query.tsv'),'w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows([[str(idx),x.replace("\r","")] for idx,x in enumerate(dev_queries)])

    with open(os.path.join(args.output_dir,'query','test_query.tsv'),'w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows([[str(idx),x.replace("\r","")] for idx,x in enumerate(test_queries)])
    
    

    
