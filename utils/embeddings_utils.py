from sklearn.manifold import TSNE
from utils.hp_request_util import hf_request
import pandas as pd


def get_embeddings_data(dataset):

    desc = [i['Description'] if i['Description'] != 'nan' else '' for i in dataset]
    tasks = [i['Tasks'] if i['Tasks'] != 'nan' else '' for i in dataset]
    abstracts = [i['Abstract'] if i['Abstract'] != 'nan' else '' for i in dataset]

    full = [' '.join(tpl) for tpl in list(zip(*[desc, tasks, abstracts]))]

    return embeddings_data(sentence_transformer_request(full))


def sentence_transformer_request(d_abstracts):
    requested_data = None
    first = 0
    last = 0

    print(len(d_abstracts))

    while last < len(d_abstracts):

        if last + 200 < len(d_abstracts):
            last += 200
        else:
            last += len(d_abstracts) - last


        response = hf_request('sentence-transformers/all-MiniLM-L6-v2', d_abstracts[first:last])

        first = last

        if requested_data is None:
            requested_data = response.json()
        else:
            requested_data += response.json()

    return pd.DataFrame(requested_data)


def embeddings_data(abstract_embeddings):

    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(abstract_embeddings)

    return pd.DataFrame(tsne_data - tsne_data.min()).to_json(orient='split'), tsne_data

