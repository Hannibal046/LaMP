# %%
from tqdm import tqdm
import json
templates = {
    "LaMP_1":r'For an author who has written the paper with the title "(.+)", which reference is related\? Just answer with \[1\] or \[2\] without explanation\. \[1\]: "(.+)" \[2\]: "(.+)"',
    "LaMP_2":r'Which category does this article relate to among the following categories\? Just answer with the category name without further explanation\. categories: \[women, religion, politics, style & beauty, entertainment, culture & arts, sports, science & technology, travel, business, crime, education, healthy living, parents, food & drink\] article: (.+)',
    "LaMP_3":r'What is the score of the following review on a scale of 1 to 5\? just answer with 1, 2, 3, 4, or 5 without further explanation\. review: (.+)',
    "LaMP_4":r'Generate a headline for the following article: (.+)',
    "LaMP_5":r'Generate a title for the following abstract of a paper: (.+)',
    "LaMP_7":r'Paraphrase the following tweet without any explanation before or after it: (.+)',
}
profile_keys = {
    "LaMP_1":["title",'abstract'],
    "LaMP_2":['text','title','category'],
    "LaMP_3":['text','score'],
    "LaMP_4":['text','title'],
    "LaMP_5":['title','abstract'],
    "LaMP_6":[],
    "LaMP_7":['text'],
}
def LaMP_remove_template(data,task):
    import re
    template = templates[task]
    ret = []
    for line in data:
        line = line.replace("\n"," ")
        match_obj = re.match(template, line)
        assert match_obj is not None
        ret.append(" ".join(match_obj.groups()))
    return ret

def LaMP_process_profiles(data,task):
    profiles = [x['profile'] for x in data]
    for profile in profiles:
        for idx in range(len(profile)):
            temp = []
            for k in profile_keys[task]:
                temp.append(profile[idx][k])
            profile[idx] = " ".join(temp)
    return profiles


# for task in ["LaMP_"+str(x) for x in [1,2,3,4,5,7]]:
#     datapath = f"../data/{task}/train_questions.json"
#     data = json.load(open(datapath))
#     data = [x['input'] for x in data]
#     data = LaMP_remove_template(data,task)
#     print(data[0])

# %%
import logging,pickle
logging.getLogger().setLevel(logging.WARNING) 
from pyserini.search.lucene import LuceneSearcher
import json,os

for task in ["LaMP_"+str(x) for x in [1,2,3,4,5,7]]:
    for _split in 'train dev'.split():
        datapath = f"../data/{task}/{_split}_questions.json"
        data = json.load(open(datapath))

        querys = LaMP_remove_template([x['input'] for x in data],task)
        documents = LaMP_process_profiles(data,task)
        output = []
        for q,d in tqdm(zip(querys,documents),total=len(querys)):
            output_per_query = []
            os.makedirs("/tmp/temp_bm25",exist_ok=True)
            with open("/tmp/temp_bm25/sample.jsonl",'w') as f:
                for idx, _d in enumerate(d):
                    to_be_dumped = {
                        "id":idx,
                        "contents":_d
                    }
                    f.write(json.dumps(to_be_dumped)+'\n')
                
            build_index_cmd = f"""python -m pyserini.index.lucene \
                --collection JsonCollection \
                --input /tmp/temp_bm25 \
                --index /tmp/temp_bm25/sample.index \
                --generator DefaultLuceneDocumentGenerator \
                --threads 1 \
                --storePositions --storeDocvectors --storeRaw >/dev/null 2>&1 """

            os.system(build_index_cmd)

            searcher = LuceneSearcher('/tmp/temp_bm25/sample.index')
            hits = searcher.search(q)
            for i in range(len(hits)):
                output_per_query.append(int(hits[i].docid))
            output.append(output_per_query)
        with open(f"../data/{task}/{_split}_bm25.pkl",'wb') as f:
            pickle.dump(output,f)


