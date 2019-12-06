from snorkel.labeling import labeling_function
from bs4 import BeautifulSoup
import codecs
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")

CATEGORY = 0
MODELNAME = 1
BRAND = 2
OTHER = 3

@labeling_function()
def lf_is_before_num(x):
    """ If the word is before a number, it is likely MODELNAME E.g. apple watch *series* 3
    """
    words = x.product_name.split()
    if x.word_idx < len(words)-1 and words[x.word_idx+1].isnumeric():
        return MODELNAME
    return -1


@labeling_function()
def lf_has_numbers(x):
    words = x.product_name.split()
    w = words[x.word_idx]
    if any(i.isdigit() for i in w):
        return MODELNAME
    return -1
    
@labeling_function()
def lf_has_digits_and_letters(x):
    words = x.product_name.split()
    w = words[x.word_idx]
    if any(i.isdigit() for i in w) and any(not i.isdigit() for i in w):
        return MODELNAME
    return -1

@labeling_function()
def lf_is_numeric(x):
    """ E.g. apple watch series *3*
    """
    words = x.product_name.split()
    if words[x.word_idx].isnumeric():
        return MODELNAME
    return -1

@labeling_function()
def lf_is_other(x):
    words = x.product_name.split()
    word = words[x.word_idx]
    # if a word ends with those measurement units, then it is likely OTHER
    if word.endswith('inch') or word.endswith('gb') or word.endswith('tb') or word.endswith('$'):
        return OTHER
    if word.startswith('$'):
        return OTHER
    if nltk.pos_tag([word])[0][1] == 'JJ':
        # if a word is an adjective, then it is likely OTHER
        return OTHER
    return -1


@labeling_function()
def lf_is_noun(x):
    words = x.product_name.split()
    word = words[x.word_idx]
    # spacey_pos = nlp(x.product_name)[x.word_idx].pos_
    spacey_pos = nlp(word)[0].pos_
    nltk_pos = nltk.pos_tag(words)[x.word_idx][1]
    if nltk_pos in ['NNS', 'NN'] and spacey_pos == 'NOUN':
        # brand names and modelnames tend be tagged as noun by pos taggers
        if all(not i.isdigit() for i in word) and word not in brand_names:
            return CATEGORY
    return -1


@labeling_function()
def lf_after_brand(x):
    words = x.product_name.split()
    if x.word_idx > 0 and x.product_name[x.word_idx-1] in brand_names:
        if any(i.isdigit() for i in words[x.word_idx]):
            return MODELNAME
        return CATEGORY
    return -1


@labeling_function()
def lf_is_single_word(x):
    words = x.product_name.split()
    if len(words) == 1:
        if words[x.word_idx] not in category_list and words[x.word_idx] not in brand_names:
            return MODELNAME
    return -1


# load external knowledge sources

# A list of categories
with open('data/categories') as f:
    category_list = f.readline().split(', ')

# Use BeautifulSoup to scrap the list of brand names from the BestBuy brands page.
f=codecs.open(r"data/NameBrands_BestBuy.html", 'r', 'utf-8')
soup= BeautifulSoup(f.read(), 'lxml')
brand_list_div = soup.find_all('div', {"class": "alphabetical-list"})[0]
brand_names = [url.get('data-lid').lower() for url in brand_list_div.find_all('a')]


@labeling_function(resources={'brand_names': brand_names})
def lf_is_brand(x, brand_names):
    """ Checks if the word is in the list of brand names.
    """
    words = x.product_name.split()
    if words[x.word_idx] in brand_names:
        return BRAND
    # check if the token is in a two word brand name
    if ' '.join(words[x.word_idx-1:x.word_idx+1]) in brand_names or ' '.join(words[x.word_idx:x.word_idx+2]) in brand_names:
        return BRAND
    return -1


@labeling_function(resources={'brand_names': brand_names})
def lf_is_after_brand(x, brand_names):
    """ If the word is after a brand name, it is likely MODELNAME. E.g. beats *solo3* wireless
    """
    words = x.product_name.split()
    if x.word_idx > 0 and words[x.word_idx-1] in brand_names:
        if any(i.isdigit() for i in words[x.word_idx]):
            return MODELNAME
        return CATEGORY
    return -1


@labeling_function(resources={'category_list': category_list})
def lf_is_in_category_list(x, category_list):
    words = x.product_name.split()
    if words[x.word_idx] in category_list or words[x.word_idx][:-1] in category_list:
        return CATEGORY
    return -1