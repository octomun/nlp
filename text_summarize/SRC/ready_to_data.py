import os
import json
from tqdm import tqdm
import re
import MeCab
import easydict
from prepro.data_builder import format_to_bert
def run(path):
    for path2 in ['train','valid']:
        DATAPATH = path+path2
        filenames = [x for x in os.listdir (DATAPATH) if x.endswith('json')]
        list_dic = []

        for file in filenames:
            filelocation = os.path.join(DATAPATH, file)

            with open(filelocation, 'r',encoding='utf-8') as json_file:
                data = json.load(json_file)['documents']
                # print(data)
                for x in tqdm (range(len(data))):
                    text = data[x]['text']
                    text = str(text).replace('"', "'")

                    extractive = data[x]['extractive']
                    for index, value in enumerate(extractive):
                        if value == None:
                            extractive[index] = 0

                    p = re.compile('(?<=sentence\'\: \')(.*?)(?=\'highlight_indices)')
                    texts = p.findall(text)

                    sentences = []
                    for t in texts:
                        sentence = t[:-3]
                        sentences.append(sentence)

                    mydict = {}
                    mydict['text'] = sentences
                    mydict['extractive'] = extractive
                    list_dic.append(mydict)
        
        with open(f"{path+path2}.json", 'w') as fh:
            json.dump(list_dic, fh)

        def list_chunk(lst, n):
            return [lst[i:i+n] for i in range(0, len(lst), n)]
        if path2 == 'train':
            with open(f"{path+path2}.json", 'r') as fh:
                data = json.load(fh)
            data_chunked = list_chunk(data, 32507) ## 전체 데이터를 10개로 분할
            for i, d in enumerate(data_chunked):
                with open(f"{path+path2}.{i}.json".format(i), 'w') as fh:
                    json.dump(d, fh)
        mecab = MeCab.Tagger()
        DATAPATH = path
        filenames = [x for x in os.listdir (DATAPATH) if path2 in x and x.endswith('json')]
        print(filenames)
        trainfiles = []
        for f in filenames[1:-1]:
            trainfiles.append(f[:-5])
        print(trainfiles)
        for set_name in trainfiles:
            print("Processing ... ", set_name)

            with open("{}/{}.json".format(path,set_name), 'r',encoding='utf-8') as f:
                data = json.load(f)

                list_dic = []
                for x in tqdm(range(len(data))):
                    text = data[x]['text']
                    extractive = data[x]['extractive']

                    sentences = []
                    for sentence in text:
                        sentence_morph = ' '.join(list(map(lambda morphs : morphs.split('\t')[0], mecab.parse(sentence).split('\n')))[:-2])
                        sentences.append(sentence_morph)

                    extractives = []
                    for e in extractive:
                        extractives.append(sentences[e])

                    src = [i.split(' ') for i in sentences]
                    tgt = [i.split(' ') for i in extractives]

                    mydict = {}
                    mydict['src'] = src
                    mydict['tgt'] = tgt
                    list_dic.append(mydict)

                jsonfilelocation = path+'../json_data/' + path2
                os.makedirs(jsonfilelocation, exist_ok=True)

                temp = []
                DATA_PER_FILE = 50

                for i,a in enumerate(tqdm(list_dic)):
                    if (i+1)%DATA_PER_FILE!=0:
                        temp.append(a)
                    else:
                        temp.append(a)
                        filename = 'korean.'+ set_name + '.' + str(i//DATA_PER_FILE)+'.json'
                        with open(os.path.join(jsonfilelocation, filename), "w", encoding='utf-8') as json_file:
                            json.dump(temp, json_file, ensure_ascii=False)
                        temp = []

                    #마지막에 남은 데이터 있으면 추가로 append
                    if len(temp) != 0:
                        filename = 'korean.'+ set_name + '.' + str(i//DATA_PER_FILE + 1)+'.json'
                        with open(os.path.join(jsonfilelocation, filename), "w", encoding='utf-8') as json_file:
                            json.dump(temp, json_file, ensure_ascii=False)
        set_name = path2

        bertfilelocation = path+'../bert_data/'+set_name
        os.makedirs(bertfilelocation, exist_ok=True)
        print(bertfilelocation, path+'../json_data/')
        args = easydict.EasyDict({
        "dataset": [set_name], 
        "raw_path": path+'../json_data/',
        "save_path": bertfilelocation,
        "n_cpus":16,
        "oracle_mode": "greedy",
        "min_src_ntokens": 5,
        "max_src_ntokens": 200,
        "min_nsents": 3,
        "max_nsents": 100,
        })
        format_to_bert(args)

if __name__ == '__main__':
    run('./raw_data/')