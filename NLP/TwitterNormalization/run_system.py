import sys
import context2vec.eval as c2v
import os
sys.path.insert(0, sys.argv[1])
from nltk.tokenize import word_tokenize

num_tweets = int(sys.argv[2])
print(type(num_tweets))
MODEL_DIR = os.path.join(os.getcwd(),'model')
MODEL_NAME = 'context2vec.ukwac.model.params'
PATH = os.path.join(MODEL_DIR,MODEL_NAME)

import time
import numpy
import traceback
import re
import six

from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader

class ParseException(Exception):
    def __init__(self, str):
        super(ParseException, self).__init__(str)

def parse_input(line):    
    target_exp = re.compile('\[.*\]')
    sent = line.strip().split()
    target_pos = None
    for i, word in enumerate(sent):
        if target_exp.match(word) != None:
            target_pos = i
            if word == '[]':
                word = None
            else:
                word = word[1:-1]
            sent[i] = word
    return sent, target_pos    

def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity<0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity<0] = 0.0
    return (target_similarity * context_similarity) 

def load_c2v_model(model_param_file):
    model_reader = ModelReader(model_param_file)
    w = model_reader.w
    word2index = model_reader.word2index
    index2word = model_reader.index2word
    model = model_reader.model
    
    model_full = [model,w,word2index,index2word]
    
    return model_full


def evalc2v(input_line, c2v_model,verbose=False,n_result=10):
    eval_list = []
#     n_result = 10  # number of search result to show
    gpu = -1 # todo: make this work with gpu
    
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()    
    xp = cuda.cupy if gpu >= 0 else numpy    
    
    try:
        line = input_line
        sent, target_pos = parse_input(line)
        if target_pos == None:
            raise ParseException("Can't find the target position.") 
                    
        if sent[target_pos] == None:
            target_v = None
        elif sent[target_pos] not in word2index:
            raise ParseException("Target word is out of vocabulary.")
        else:
            target_v = w[word2index[sent[target_pos]]]
        if len(sent) > 1:
            context_v = c2v_model[0].context2vec(sent, target_pos) 
            context_v = context_v / xp.sqrt((context_v * context_v).sum())
        else:
            context_v = None        
            
        if target_v is not None and context_v is not None:
            similarity = mult_sim(c2v_model[1], target_v, context_v)
        else:
            if target_v is not None:
                v = target_v
            elif context_v is not None:
                v = context_v                
            else:
                raise ParseException("Can't find a target nor context.")   
            similarity = (c2v_model[1].dot(v)+1.0)/2 # Cosine similarity can be negative, mapping similarity to [0,1]
                
        count = 0
        for i in (-similarity).argsort():
            if numpy.isnan(similarity[i]):
                continue
            eval_list.append((c2v_model[3][i], similarity[i]))
            
            if verbose is True:
                print('{0}: {1}'.format(c2v_model[3][i], similarity[i]))
            count += 1
            if count == n_result:
                break

    except ParseException as e:
        print "ParseException: {}".format(e)                
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print "*** print_tb:"
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
    return eval_list

def computeEditDistance(str1, str2,w_del=1,w_ins=1,w_sub=1):
    str1 = str1.lower()
    str2 = str2.lower()
    m = numpy.zeros((len(str1)+1,len(str2)+1))
    for i in range(len(str1)+1):
        m[i,0] = i
    for j in range(len(str2)+1):
        m[0,j] = j
    for i in range(len(str1)+1)[1:]:
        for j in range(len(str2)+1)[1:]:
            if str1[i-1] == str2[j-1]:
#                 m[i,j] = min(m[i-1,j]+w_del, m[i,j-1]+w_ins, m[i-1,j-1])
                m[i,j] = m[i-1,j-1]
            else:
                m[i,j] = min(m[i-1,j]+w_del, m[i,j-1]+w_ins, m[i-1,j-1]+w_sub)
    return m[len(str1),len(str2)]

def read_dataset(dataset):
    text = []
    with open(dataset, "r") as f:
        text = f.readlines()
        
    for i,line in enumerate(text):
        text[i] = line.split('\n')[0]
    return text

def write_output(filename,text_list):
    with open(filename, "w") as f:
        for tweet in text_list:
            f.write(tweet+'\n')


dict_abbreviation = {'a':{'atm':'at the moment',
                          'af':'as fuck',
                          'afk':'away from keyboard'},
                     'b':{'b':'be',
                          'bc':'because',
                          'brb':'be right back',
                          'bro':'friend',
                          'bb':'baby',
                          'bm':'bad manners',
                          'bs':'bullshit',
                          'bbq':'barbecue'},
                     'c':{'cya':'see you later',
                          'cu':'see you later',
                          'cus':'because',
                         },
                     'd':{'dis':'this',
                          'dat':'that',
                          'dawg':'friend',
                          'dafuq':'what the fuck',
                          'dm':'direct message',
                          'dang':'wow'},
                     'e':{'ez':'easy'
                         },
                     'f':{'fu':'fuck you',
                          'ffs':'for fuck sake',
                          'fr':'for real',
                          'fml':'fuck my life'},
                     'g':{'gov':'government',
                          'gg':'good game',
                          'ggla':'good game, love all',
                          'gr8':'great',
                          'gtfo':'go away',
                          'gth':'go to hell',
                          'gtg':'have to go'
                         },
                     'h':{},
                     'i':{'ikr':'I know, right?',
                          'idc':'I do not care',
                          'idgaf':'I do not care',
                          'immo':'in my modest opinion',
                          'iirc':'if I remember correctly'
                         },
                     'j':{},
                     'k':{'k':'okay',
                          'kk':'okay'
                         },
                     'l':{'lmao':'*laugh',
                          'lmfao':'*laugh*',
                          'lol':'*laugh*'
                         },
                     'm':{'m8':'mate',
                          'mf':'motherfucker'
                         },
                     'n':{'nsfw':'not safe for work',
                          'nbd':'no big deal'},
                     'o':{'ok':'okay',
                          'omg':'oh my god',
                          'omfg':'oh my fucking god'},
                     'p':{'pm':'private message',
                          'ppl':'people',
                          'plz':'please'
                         },
                     'q':{},
                     'r':{'r':'are',
                          'rofl':'*laugh*',
                          'rn':'right now',
                         'ru':'are you'},
                     's':{'smh':'shaking my head',
                          'sis':'friend',
                          'sry':'sorry',
                          'stfu':'shut the fuck up',
                          'smth':'something'
                         },
                     't':{'ttyl':'talk to you later',
                          'tn':'tonight'},
                     'u':{'u':'you',
                          'ur':'your'},
                     'v':{},
                     'w':{'wtf':'what the fuck',
                          'wth':'what the hell'},
                     'x':{'xoxo':'kissing and hugging',
                          'xo':'kissing',
                          'xx':'kissing'},
                     'y':{'yolo':'you only live once'},
                     'z':{}
                    }

def abcTo123():
    alphabet_dict = {'a':0,
                 'b':1,
                 'c':2,
                 'd':3,
                 'e':4,
                 'f':5,
                 'g':6,
                 'h':7,
                 'i':8,
                 'j':9,
                 'k':10,
                 'l':11,
                 'm':12,
                 'n':13,
                 'o':14,
                 'p':15,
                 'q':16,
                 'r':17,
                 's':18,
                 't':19,
                 'u':20,
                 'v':21,
                 'w':22,
                 'x':23,
                 'y':24,
                 'z':25}
    
    return alphabet_dict

def build_indexed_alphabet(word_dict_txt):
    large_dict = []
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    for letter in alphabet:
        list_letter = [word for word in word_dict_txt if word.startswith(letter)]
        large_dict.append(list_letter)
        
    return large_dict    

# Removing non ASCII elements

def remove_no_ascii(text):
    import re
    text = re.sub(r'[^\x00-\x7F]+','', text)
    text = re.sub(r'&amp;',r'and',text)
    text = re.sub(r'&gt;',r'>',text)
    text = re.sub(r'&lt;',r'<',text)
    text = re.sub(r'\'',r"'",text)
    return text

# Removing RTs

def clear_RTs(text):
    clean_text = text
    idx = clean_text.find('RT')

    while idx != -1:    
        idx_end = clean_text.find(': ',idx)+1
        if idx_end == 0:
            idx_end = len(clean_text)
        RT = clean_text[idx:idx_end]
        clean_text = clean_text.replace(RT, '')
        idx = clean_text.find('RT')
        
    return clean_text

# Removing URLs

def clear_URLs(text):
    clean_text = text
    idx = clean_text.find('http')

    while idx != -1:    
        idx_end = clean_text.find(' ',idx)
        if idx_end == -1:
            idx_end = len(clean_text)
        url = clean_text[idx:idx_end]
        clean_text = clean_text.replace(url, '')
        idx = clean_text.find('http')
        
    return clean_text

def clear_At(text):    
    clean_text = re.sub(r'@[a-zA-Z0-9_]{1,15}','',text)               

    return clean_text

# Removing HTML

def clear_HTMLs(text):    
    text = re.sub(r'<[a-zA-Z0-9]>','',text)
    text = re.sub(r'</[a-zA-Z0-9]>','',text)
    return text

def parseHashtags(text):            
    clean_text = text
    idx = clean_text.find('#')
    
    while idx != -1:
        idx_end = 1
        if idx!=len(clean_text):
            while idx+idx_end<len(clean_text):
                if not clean_text[idx+idx_end].isalpha() and not clean_text[idx+idx_end].isdigit():
                    break
                else:
                    idx_end += 1
                    
            hashtag = clean_text[idx:idx+idx_end]
#             print(hashtag)
            hashtag = re.sub(r'#([a-zA-Z0-9])',r' #\1',hashtag)   
#             print(hashtag)
            hashtag = re.sub(r'([a-zA-Z])([0-9])([a-zA-Z])',r'\1-\2-\3',hashtag)
#             print(hashtag)
            hashtag = re.sub(r'([0-9])([a-zA-Z])',r'\1-\2',hashtag)     
#             print(hashtag)            
            hashtag = re.sub(r'([a-z])([A-Z0-9])',r'\1-\2',hashtag)
#             print(hashtag)
            hashtag = re.sub(r'([A-Z]{1})([A-Z]{1})([a-z0-9])',r'\1-\2\3',hashtag)

        clean_text = clean_text[:idx]+hashtag+clean_text[idx+idx_end:]
        idx = clean_text.find('#',idx+len(hashtag))

    return clean_text

def correctSlang(text,dictionnary):
    clean_text = text
    split = re.split("-|;|:|\(|\)|\[|\]|\.| |,|!|\~|\?",text)
    parsed_tweet = [s for s in split if len(s)!=0]
    
    for word in parsed_tweet:
        word = word.lower()
        idx = 0
        if len(word)==1 and word[0].isalpha():
            if word in dictionnary[word.lower()]:
                idx = clean_text.lower().find(word,idx)
                while idx!=-1:                    
                    if idx<len(clean_text)-1:
                        if clean_text[idx-1].isalpha() or clean_text[idx+1].isalpha():
                            idx = clean_text.lower().find(word,idx+1)
                        else:
                            clean_text = clean_text[:idx]+dictionnary[word[0].lower()][word]+clean_text[idx+1:]
                            idx = clean_text.lower().find(word,idx+len(dictionnary[word[0].lower()][word]))
                    elif idx==len(clean_text)-1:
                        if not clean_text[idx-1].isalpha():
                            clean_text = clean_text[:idx]+dictionnary[word[0].lower()][word]
                        idx=-1
                        
                    elif idx==0:
                        if not clean_text[idx+1].isalpha():
                            clean_text = dictionnary[word.lower()][word]+clean_text[1:]
                            idx = clean_text.lower().find(word,idx+len(dictionnary[word.lower()][word]))
                    
                idx = 0
        
        elif len(word)>1 and word[0].isalpha():
            if word in dictionnary[word[0].lower()]:
                idx = clean_text.lower().find(word,idx)
                while idx!=-1:                     
                    if idx>0:                        
                        if idx<len(clean_text)-len(word):
                            if clean_text[idx-1].isalpha() or clean_text[idx+len(word)].isalpha():
                                idx = clean_text.lower().find(word,idx+1)
                                
                            else:
                                clean_text = clean_text[:idx]+dictionnary[word[0].lower()][word]+clean_text[idx+len(word):]
                                idx = clean_text.lower().find(word,idx+len(dictionnary[word[0].lower()][word]))
                                
                        elif idx==len(clean_text)-len(word):
                            if not clean_text[idx-1].isalpha():
                                clean_text = clean_text[:idx]+dictionnary[word[0].lower()][word]
                            idx=-1
                                
                    elif idx==0:
                        if idx+len(word)<len(clean_text):
                            if not clean_text[idx+len(word)].isalpha():
                                clean_text = dictionnary[word[0].lower()][word]+clean_text[len(word):]
                            idx = clean_text.lower().find(word,idx+len(dictionnary[word[0].lower()][word]))
                        else:
                            idx=-1
                idx = 0
                
    return clean_text
            
def remove_verb_contractions(text):
    clean_text = text
    # TODO remplacer while--> idx = clean... par une function(text,str)
    idx = clean_text.lower().find("'s")
    while idx != -1 and idx!=1:
        pre_letter = clean_text[idx-2]+clean_text[idx-1]
        if idx<len(clean_text)-len("'s"):
            post_letter = clean_text[idx+len("'s")]
        else:
            post_letter = '.'
            
        if pre_letter.lower() in ['he','it','at'] and not post_letter.isalpha():
            
            idx_end = idx+len("'s")                
            word = clean_text[idx:idx_end]
            if word.upper()==word:
                clean_text = clean_text[:idx] + " 'S " + clean_text[idx+len("'s")+1:]#clean_text[idx:].replace(word, " 'S", 1)
            else:
                clean_text = clean_text[:idx] + " 's " + clean_text[idx+len("'s")+1:]
        idx = clean_text.lower().find("'s",idx+1,len(clean_text))
        
    idx = clean_text.lower().find("'d")
    while idx != -1 and idx!=1:
        pre_letter = clean_text[idx-2]+clean_text[idx-1]
        if idx<len(clean_text)-len("'d"):
            post_letter = clean_text[idx+len("'d")]
        else:
            post_letter = '.'
            
        if pre_letter.lower() in ['he','it','at','ou'] and not post_letter.isalpha():
            
            idx_end = idx+len("'d")                
            word = clean_text[idx:idx_end]
            if word.upper()==word:
                clean_text = clean_text[:idx] + " 'D " + clean_text[idx+len("'D")+1:]#clean_text[idx:].replace(word, " 'S", 1)
            else:
                clean_text = clean_text[:idx] + " 'd " + clean_text[idx+len("'d")+1:]
        idx = clean_text.lower().find("'d",idx+1,len(clean_text))
    
    idx = clean_text.lower().find("'re")
    while idx != -1 and idx!=1:
        pre_letter = clean_text[idx-1]
        if idx<len(clean_text)-len("'re"):
            post_letter = clean_text[idx+len("'re")]
        else:
            post_letter = '.'
        if pre_letter.lower() in ['u','y','e'] and not post_letter.isalpha():
            
            idx_end = idx+len("'re")                
            word = clean_text[idx:idx_end]
            if word.upper()==word:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' ARE', 1)
            else:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' are', 1)

        idx = clean_text.lower().find("'re",idx+1,len(clean_text))
        
    idx = clean_text.lower().find("'ve")
    while idx != -1 and idx!=1:
        pre_letter = clean_text[idx-1]
        if idx<len(clean_text)-len("'ve"):
            post_letter = clean_text[idx+len("'ve")]
        else:
            post_letter = '.'
        if pre_letter.lower() in ['i','u','y','e','t'] and not post_letter.isalpha():
             
            idx_end = idx+len("'ve")                
            word = clean_text[idx:idx_end]
            if word.upper()==word:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' HAVE', 1)
            else:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' have', 1)

        idx = clean_text.lower().find("'ve",idx+1,len(clean_text))
        
    idx = clean_text.lower().find("'ll")
    while idx != -1 and idx!=1:
        pre_letter = clean_text[idx-1]
        if idx<len(clean_text)-len("'ll"):
            post_letter = clean_text[idx+len("'ll")]
        else:
            post_letter = '.'
        if pre_letter.lower() in ['i','u','y','e','t'] and not post_letter.isalpha():
            
            idx_end = idx+len("'ll")            
            word = clean_text[idx:idx_end]
            
            if word.upper()==word:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' WILL', 1)
            else:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' will', 1)

        idx = clean_text.lower().find("'ll",idx+1,len(clean_text))
    
        
    idx = clean_text.lower().find("n't")
    while idx != -1 and idx!=1:
        pre_letter = clean_text[idx-1]
        if idx<len(clean_text)-len("n't"):
            post_letter = clean_text[idx+len("n't")]
        else:
            post_letter = '.'
        if pre_letter.lower() in ['o','s','a','e','d'] and not post_letter.isalpha():
        
            idx_end = idx+len("n't")    
            word = clean_text[idx:idx_end]
            
            if clean_text.lower()[idx-1] != 'a':
                if word.upper()==word:
                    clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' NOT', 1)
                else:
                    clean_text = clean_text[:idx] + clean_text[idx:].replace(word, ' not', 1)
            else:
                if word.upper()==word:
                    clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'NNOT', 1)
                else:
                    clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'nnot', 1)
            
        idx = clean_text.lower().find("n't",idx+1,len(clean_text))
        
    idx = clean_text.lower().find("gonna")
    while idx != -1:
        if idx!=1:
            pre_letter = clean_text[idx-1]
        else:
            pre_letter = '.'
        if idx<len(clean_text)-len("gonna"):
            post_letter = clean_text[idx+len("gonna")]
        else:
            post_letter = '.'
        if not pre_letter.isalpha() and not post_letter.isalpha():
            
            idx_end = idx+len("gonna")            
            word = clean_text[idx:idx_end]
            
            if word.upper()==word:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'GOING TO', 1)
            else:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'going to', 1)

        idx = clean_text.lower().find("gonna",idx+1,len(clean_text))    
        
        
    idx = clean_text.lower().find("wanna")
    while idx != -1:
        if idx!=1:
            pre_letter = clean_text[idx-1]
        else:
            pre_letter = '.'
        if idx<len(clean_text)-len("wanna"):
            post_letter = clean_text[idx+len("wanna")]
        else:
            post_letter = '.'
        if not pre_letter.isalpha() and not post_letter.isalpha():
            
            idx_end = idx+len("wanna")            
            word = clean_text[idx:idx_end]
            
            if word.upper()==word:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'WANT TO', 1)
            else:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'want to', 1)

        idx = clean_text.lower().find("wanna",idx+1,len(clean_text))  
        
    idx = clean_text.lower().find("gotta")
    while idx != -1:
        if idx!=1:
            pre_letter = clean_text[idx-1]
        else:
            pre_letter = '.'
        if idx<len(clean_text)-len("gotta"):
            post_letter = clean_text[idx+len("gotta")]
        else:
            post_letter = '.'
        if not pre_letter.isalpha() and not post_letter.isalpha():
            
            idx_end = idx+len("gotta")            
            word = clean_text[idx:idx_end]
            
            if word.upper()==word:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'GOT TO', 1)
            else:
                clean_text = clean_text[:idx] + clean_text[idx:].replace(word, 'got to', 1)

        idx = clean_text.lower().find("gotta",idx+1,len(clean_text))  
            
    clean_text = re.sub(r'([iI]{1})\'[Mm]',r'\1 am', clean_text)
    clean_text = re.sub(r'I\'M',r'I AM', clean_text)
        
    return clean_text

def remove_ends(text):
    clean_text = ''
    j = 0
    while j<len(text) and text[j].isalpha() is False:
        j=j+1
    clean_text = text[j:]
    
    j=1
    while True:
        if len(clean_text)>1:
            if not clean_text[-j].isalpha() and not clean_text[-j-1].isalpha():
                clean_text = clean_text[:-j-1]
            elif clean_text[-j].isalpha() and clean_text[-j-1].isalpha():
                break
            else:
                clean_text = clean_text[:-j]
        else:
            break
    
#     if clean_text[-1].isalpha()

    return clean_text

def use_dict2vec(tweet, c2v_model,indexed_word_dict,w_del=2,w_ins=0.5,w_sub=1,nb_c2v=100,verbose=False):
    split = re.split("-|;|:|\(|\)|\[|\]|\.| |,|!|\~|\?",tweet)
    parsed_tweet = [s for s in split if len(s)!=0]
    alphabet_dict = abcTo123()
    if len(parsed_tweet)>1:
        for word in parsed_tweet:
            if word[0].isalpha() and word[0].islower():
                dict_letter = indexed_word_dict[alphabet_dict[word[0].lower()]]
                
                if word.lower() not in dict_letter:
                    i_start = 0
                    w = [w_del,w_ins,w_sub]
                    stop = False
                    while stop is False:                        
                        idx = tweet.find(word,i_start)
                        if idx>0 and idx<len(tweet)-len(word):
                            if not tweet[idx-1].isalpha() and not tweet[idx+len(word)].isalpha():
                                tweet2pred = tweet[:idx].lower()+' [] '+tweet[idx+len(word):].lower()
                                eval_vec = evalc2v(tweet2pred,c2v_model,False,nb_c2v)
                                stop=True
                            else:
                                i_start = idx+1
                        elif idx==0 and len(tweet)>1:
                            if not tweet[idx+len(word)].isalpha():
                                tweet2pred = tweet[:idx].lower()+' [] '+tweet[idx+len(word):].lower()
                                # we put everything on lower case to use it with c2v
                                eval_vec = evalc2v(tweet2pred,c2v_model,False,nb_c2v)
                                stop=True
                            else:
                                i_start = idx+1
                                
                        elif idx==len(tweet)-len(word):
                            if not tweet[idx-1].isalpha():
                                tweet2pred = tweet[:idx].lower()+' [] '+tweet[idx+len(word):].lower()
                                eval_vec = evalc2v(tweet2pred,c2v_model,False,nb_c2v)
                                w = [2,1,1]
                                stop=True
                            else:
                                i_start = idx+1
                        else:
                            i_start = idx+1
                            
                    if verbose is True:
                        print(tweet2pred)
                    dist = 100
                    idx_min = 0
                    
                    candidates = [e[0] for e in eval_vec if e[0].isalpha()]+dict_letter
                                        
                    for i,candidate in enumerate(candidates):
                        if candidate[0].isalpha():
                            dist_temp = computeEditDistance(word.lower(), candidate.lower(),w[0],w[1],w[2])
                            if i<len(eval_vec):
                                dist_temp = dist_temp*(1/eval_vec[i][1])*0.4
                                # This lowers the distance of words predicted with context2vec, with most probable words having closest distance
                                # if the prediction has a probability <0.4, the distance is then amplified by a factor 0.4/probability
#                             computeEditDistance(str1, str2,w_del=1,w_ins=1,w_sub=1):
                            if dist_temp<dist:
                                dist=dist_temp
                                idx_min = i
                
                    tweet = tweet[:idx]+candidates[idx_min]+tweet[idx+len(word):]
                    
            elif word.lower()=="'s" or word.lower()=="'d":
                i_start = 0
                stop = False
                while stop is False:
                    idx = tweet.find(word,i_start)
                    if idx>0 and idx<len(tweet)-len(word):
                        if not tweet[idx-1].isalpha() and not tweet[idx+len(word)].isalpha():
                            tweet2pred = tweet[:idx].lower()+' [] '+tweet[idx+len(word):].lower()
                            eval_vec = evalc2v(tweet2pred,c2v_model,False,10)
                            stop=True
                        else:
                            i_start = idx+1
                    elif idx==0 and len(tweet)>1:
                        if not tweet[idx+len(word)].isalpha():
                            tweet2pred = tweet[:idx].lower()+' [] '+tweet[idx+len(word):].lower()
                            eval_vec = evalc2v(tweet2pred,c2v_model,False,10)
                            stop=True
                        else:
                            i_start = idx+1
                            
                    elif idx==len(tweet)-len(word):
                        if not tweet[idx-1].isalpha():
                            tweet2pred = tweet[:idx].lower()+' [] '+tweet[idx+len(word):].lower()
                            eval_vec = evalc2v(tweet2pred,c2v_model,False,10)
                            
                            stop=True
                        else:
                            i_start = idx+1
                    else:
                        i_start = idx+1
                        
                if verbose is True:
                    print(tweet2pred)
                dist = 100
                idx_min = 0
                
                candidates = [e[0] for e in eval_vec if e[0].isalpha()]
                for i,candidate in enumerate(candidates):
                    if candidate[0].isalpha():
                        dist_temp = computeEditDistance(word.lower(),candidate.lower(),1000,1,1000)
                        
                        if dist_temp<dist and dist_temp<5:
                            dist=dist_temp
                            idx_min = i
                            
                tweet = tweet[:idx]+candidates[idx_min]+tweet[idx+len(word):]

                
    else:
        w = [2,1,2]
        if tweet[0].isalpha() and tweet[0].islower():
            dict_letter = large_dict[alphabet_dict[tweet[0].lower()]]
            if tweet not in dict_letter:
                dist = 100
                idx_min = 0
                for i, candidate in enumerate(dict_letter):
                    dist_temp = computeEditDistance(tweet.lower(),candidate.lower(),w[0],w[1],w[2])
                    if dist_temp<dist:
                        dist=dist_temp
                        idx_min = i
                        
                tweet = dict_letter[idx_min]
            
    return tweet

def clean_homophones(tweet,c2v_model):
    split = re.split("-|;|:|\(|\)|\[|\]|\.| |,|!|\~|\?",tweet)
    parsed_tweet = [s for s in split if len(s)!=0]
    idx_start = 0
    
    word = parsed_tweet[0]
    if word[0].islower():
        tweet2pred = ' [] '+tweet[len(parsed_tweet[0]):]
        eval_vec = evalc2v(tweet2pred,c2v_model,False,4)
        dist = 100
        idx_min = -1
        candidates = [ev[0] for ev in eval_vec if ev[0].isalpha()]
        for i,e in enumerate(candidates):
            dist_temp = computeEditDistance(word.lower(), e,1,1,1)
            if dist_temp<2 and dist_temp<dist:
                dist=dist_temp
                idx_min = i
                
        if idx_min != -1:          
            tweet = candidates[idx_min]+' '+tweet[len(word):]    
            idx_start += len(candidates[idx_min])+1
            
        idx_start = len(word)+1
    else:
        idx_start = len(word)+1
        
    for word in parsed_tweet[1:]:
        if word[0].islower():
            idx_start_updated = tweet.find(word, idx_start)
            tweet2pred = tweet[:idx_start_updated]+' [] '+tweet[idx_start_updated+len(word):]
            eval_vec = evalc2v(tweet2pred,c2v_model,False,4)
            dist = 100
            idx_min = -1
            candidates = [ev[0] for ev in eval_vec if ev[0].isalpha()]
            for i,e in enumerate(candidates):
                dist_temp = computeEditDistance(word.lower(), e,1,1,1)
                if dist_temp<2 and dist_temp<dist:
                    dist=dist_temp
                    idx_min = i
                    
            if idx_min !=-1:               
                tweet = tweet[:idx_start_updated]+candidates[idx_min]+tweet[idx_start_updated+len(word):]  
                idx_start += len(candidates[idx_min])+1
            else:
                idx_start += len(word)+1
        else:
            idx_start += len(word)+1
                       
    return tweet

def normalize_tweet_text(text,c2v_model,wordDict,slangDict,verbose=False):    
    normalized_text = []
    tASCII = 0
    tRT = 0
    tURLs = 0
    tHash = 0
    tAt = 0
    tEnds = 0
    tVerb = 0
    tSlang = 0
    tD2V = 0
    tHomo = 0
    
    print('Printing the following\n 1. Original tweet\n 2. Cleaned for tokenization\n 3. After tokenization\n\n\n')
    
    for i,tweet in enumerate(text):
        if len(tweet)!=0:  
            print('\n\n')
            print(tweet)
            t = time.time()
            tweet = remove_no_ascii(tweet)
            tASCII = tASCII + time.time()-t
            
            t = time.time()
            tweet = clear_RTs(tweet)
            tRT = tRT + time.time()-t
            
            t = time.time()
            tweet = clear_URLs(tweet)
            tweet = clear_HTMLs(tweet)
            tURLs = tURLs + time.time()-t
            
            t = time.time()
            tweet = parseHashtags(tweet)
            tHash = tHash + time.time()-t
            
            t = time.time()
            tweet = clear_At(tweet)
            tAt = tAt + time.time()-t
            
            t = time.time()
            tweet = remove_ends(tweet)
            tEnds = tEnds + time.time()-t
            
            t = time.time()
            tweet = remove_verb_contractions(tweet)
            tVerb = tVerb + time.time()-t
            
            
            t = time.time()
            tweet = correctSlang(tweet,slangDict)
            tSlang = tSlang + time.time()-t
            
            
            # Now using context2vec and Edit distance
        if len(tweet)!=0:
            t = time.time()
            tweet = use_dict2vec(tweet, c2v_model,wordDict,1,1,1,400)
            tD2V = tD2V + time.time()-t
            
            t = time.time()
            tweet = clean_homophones(tweet, c2v_model)
            tHomo = tHomo + time.time()-t
            
            print(tweet)
            
        if len(tweet)!=0:
#             normalized_text.append(tweet)            
            tweet = word_tokenize(tweet)
            print(tweet)
            normalized_text.append(tweet)
    
        if i>0 and i%100000==0:
            if verbose is True:
                print(i)
                print('Time to clear non ASCII characters: '+str(tASCII)+'s')
                print('Time to clear RTs: '+str(tRT)+'s')
                print('Time to clear URLs and HTML '+str(tURLs)+'s')
                print('Time to clear Hashtags: '+str(tHash)+'s')
                print('Time to clear ATs: '+str(tAt)+'s')
                print('Time to remove Ends: '+str(tEnds)+'s')
                print('Time to clean verbs and negation: '+str(tVerb)+'s')
                print('Time to clean slang: '+str(tSlang)+'s')
                print('Time to clean misspelled words: '+str(tD2V)+'s')
                print('Time to clean common homophonic errors: '+str(tHomo)+'s')
                
            tASCII = 0
            tRT = 0
            tURLs = 0
            tHash = 0
            tAt = 0
            tEnds = 0
            tVerb = 0
            tSlang = 0
            tD2V = 0

    return normalized_text

def main():
    print('Importing the context2vec model')
    c2v_model = load_c2v_model(PATH)
    print('Done')
    print('Reading the corpus')
    dataset = 'Corpus/CorpusBataclan_en.1M.raw.txt'
    text_full = read_dataset(dataset)
    print('Done')
    print('Constructing the word dictionnary')
    large_dict_txtfile = read_dataset('large.txt')
    large_dict = build_indexed_alphabet(large_dict_txtfile)
    print('Done')
    
    import random 
    rand_vec = random.sample(range(0,len(text_full)-1),num_tweets)
    
    text_sample = []
    for val in rand_vec:
        text_sample.append(text_full[val])
    
    if num_tweets<2:
        print('num_tweets must greater or equal to 2')
    else:
        norm_tweets = normalize_tweet_text(text_sample,c2v_model,large_dict,dict_abbreviation,False)
    

main()
    
    
    