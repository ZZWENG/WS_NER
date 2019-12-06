import pandas as pd

start_tok, end_tok = '[S]', '[E]'
to_tokens = lambda s: [start_tok] + s.split() + [end_tok]
label_list = ['category', 'modelname', 'brand', 'other']
label_to_int = lambda label: label_list.index(label)

def preprocess_data(data_split):
    df_data = []
    for name, annotations in data_split.items():
        toks = to_tokens(name)
        ngrams = list(zip(*[toks[i:] for i in range(3)]))
        for i, ngram in enumerate(ngrams):
            token_to_tag = ngram[1]
            if token_to_tag not in annotations:
                label = label_list.index('other')
            else:
                label = label_list.index(annotations.get(ngram[1]))
            df_data.append({'product_name': name, 'word_idx': i, #'feature': ngram, 
                            'label': label})
    df = pd.DataFrame(df_data)
    return df


def get_tag(df, product, word_idx, label='label'):
    df = df[df.product_name == product]
    tag_id = df[df.word_idx == word_idx][label].values[0]
    return label_list[tag_id]


def build_seq_tag_df(df, label='label'):
    product_tags = []
    for product in df.product_name.unique():
        tokens = product.split()
        tags = [get_tag(df, product, i, label) for i in range(len(tokens))]
        product_tags.append((product, ' '.join(tags)))

    seq_df = pd.DataFrame(product_tags, columns=['product', 'label'])
    return seq_df
