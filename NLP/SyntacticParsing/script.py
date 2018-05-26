import numpy as np
from nltk import Tree
import os
import nltk

TREEPATH = os.path.join(os.getcwd(),'sequoia-corpus+fct.mrg_strict')


def erase_pattern(pattern, string):
    idx=0
    while idx!=-1:
        idx = string.find(pattern)
        

import re

def convert_to_CNF(corpus_text):
    corpus_CNF = []
    for t in corpus_text:
        # set to Chomsky Normal Format
        tr = Tree.fromstring(t)
        tr.chomsky_normal_form()
        tr_rules = tr.productions() 
#         print(tr_rules)
        for i,rule in enumerate(tr_rules):
            tr_rules[i] = str(rule).split(' -> ')
            for j,part in enumerate(tr_rules[i]):
                tr_rules[i][j] = re.sub(r'\|.*$','',tr_rules[i][j])
                
        corpus_CNF.append(tr_rules)
        
    return corpus_CNF
    


def extractPCFG(corpus_train, corpus_dev, corpus_test):
    # We add the terminal nodes of the test set to the grammar
    
    grammar_parents = []
    grammar_daughters = []
    probas = []
    for SENT in corpus_train:
        for rule in SENT:
            if rule[0] not in grammar_parents:
                grammar_parents.append(rule[0])
                grammar_daughters.append([])
                grammar_daughters[-1].append(rule[1])
                probas.append([])
                probas[-1].append([])
                probas[-1][-1] = 1.0
                
            else:
                idx_parent = grammar_parents.index(rule[0])
                if rule[1] not in grammar_daughters[idx_parent]:
                    grammar_daughters[idx_parent].append(rule[1])
                    probas[idx_parent].append([])
                    probas[idx_parent][-1] = 1.0
                else:
                    probas[idx_parent][grammar_daughters[idx_parent].index(rule[1])] += 1.0

    for SENT in corpus_test:
        for rule in SENT:
            if rule[0] in grammar_parents:
                idx_parent = grammar_parents.index(rule[0])
                if rule[1].startswith("'") or rule[1].startswith('"'):
                    if rule[1] not in grammar_daughters[idx_parent]:
                        grammar_daughters[idx_parent].append(rule[1])
                        probas[idx_parent].append([])
                        probas[idx_parent][-1] = 1.0
                    else:
                        probas[idx_parent][grammar_daughters[idx_parent].index(rule[1])] += 1.0
                    
    for SENT in corpus_dev:
        for rule in SENT:
            if rule[0] in grammar_parents:
                idx_parent = grammar_parents.index(rule[0])
                if rule[1].startswith("'") or rule[1].startswith('"'):
                    if rule[1] not in grammar_daughters[idx_parent]:
                        grammar_daughters[idx_parent].append(rule[1])
                        probas[idx_parent].append([])
                        probas[idx_parent][-1] = 1.0
                    else:
                        probas[idx_parent][grammar_daughters[idx_parent].index(rule[1])] += 1.0             
            
    print(grammar_daughters[1])
    for i,parent_daughters in enumerate(probas):
        nb_parent_daughters = sum(parent_daughters)*1.0
        for j,daughter in enumerate(parent_daughters):
            probas[i][j] /= nb_parent_daughters                    
            
    
    return grammar_parents,grammar_daughters,probas

#Version which handles unary rules
def extractPCFGu(corpus_train, corpus_dev, corpus_test):
    # We add the terminal nodes of the test set to the grammar
    
    grammar_parents = []
    grammar_daughters = {}
    probas = {}
    for SENT in corpus_train:
        for rule in SENT:
            if rule[0] not in grammar_parents:
                grammar_parents.append(rule[0])
                grammar_daughters[rule[0]] = []
                grammar_daughters[rule[0]].append(rule[1])
                probas[rule[0]] = {rule[1] : 1.0}   
            else:
                if rule[1] not in grammar_daughters[rule[0]]:
                    grammar_daughters[rule[0]].append(rule[1])
                    probas[rule[0]][rule[1]] = 1.0
                else:
                    probas[rule[0]][rule[1]] += 1.0
            

    for SENT in corpus_test:
        for rule in SENT:
            if rule[0] in grammar_parents:
                if rule[1].startswith("'") or rule[1].startswith('"'):
                    if rule[1] not in grammar_daughters[rule[0]]:
                        grammar_daughters[rule[0]].append(rule[1])
                        probas[rule[0]][rule[1]] = 1.0
                    else:
                        probas[rule[0]][rule[1]] += 1.0
                    
    for SENT in corpus_dev:
        for rule in SENT:
            if rule[0] in grammar_parents:
                if rule[1].startswith("'") or rule[1].startswith('"'):
                    if rule[1] not in grammar_daughters[rule[0]]:
                        grammar_daughters[rule[0]].append(rule[1])
                        probas[rule[0]][rule[1]] = 1.0
                    else:
                        probas[rule[0]][rule[1]] += 1.0  
                        
    for i,parent in enumerate(probas):
        nb_parent_daughters = sum([probas[parent][r] for r in probas[parent]])
        
        for daughter in probas[parent]:
            
            probas[parent][daughter] /= nb_parent_daughters                    
            
    return grammar_parents,grammar_daughters,probas
    
    
def P_CKY(words,grammar,probas):
    table = np.zeros((len(words)+1,len(words)+1,len(grammar['parents'])))
    back = np.zeros((len(words)+1,len(words)+1,len(grammar['parents']),3))
    for j in range(len(words)+1)[1:]:
        for a,A in enumerate(grammar['parents']):
            daughters_list = grammar['daughters'][a]
            if words[j-1] in daughters_list:
                table[j-1,j,a] = probas[a][daughters_list.index(words[j-1])]
        for i in range(j-2,-1,-1):
            for k in range(i+1,j):
                #------- for all {A|A->BC in grammar} ------
                for a, A in enumerate(grammar['parents']):
                    for bc, BC in enumerate(grammar['daughters'][a]):
                        if len(BC.split(' '))==2: # Check that BC is actually not B 
                            b = grammar['parents'].index(BC.split(' ')[0])   
                            c = grammar['parents'].index(BC.split(' ')[1])
                            if table[i,k,b]>0 and table[k,j,c]>0: #-- and table[i,k,B]>0 and table[k,j,C]>0---
                                #------- for all {A|A->BC in grammar}--------
                                PROB = probas[a][bc]*table[i,k,b]*table[k,j,c]
                                if table[i,j,a]<PROB:
                                    table[i,j,a] = PROB
                                    back[i,j,a] = [k,b,c]
    
    return back, table
    
#Version which handles unary rules
def P_CKY_u(words,grammar,probas):
    table = {}    
    back = {}
    for A in grammar['parents']:
        table[A] = np.zeros((len(words)+1,len(words)+1))
        back[A] = np.empty((len(words)+1,len(words)+1),dtype=object)
    
        for i in np.indices((len(words)+1,len(words)+1)):
            back[A][i] = {}

    for j in range(len(words)+1)[1:]:
        print('word '+str(j))
        for a,A in enumerate(grammar['parents']):
            daughters_list = grammar['daughters'][A]
            if words[j-1] in daughters_list:
                table[A][j-1,j] = probas[A][words[j-1]]
                
            added = True
            while added:
                added = False
                for b,B in enumerate(grammar['parents']):
                    if table[B][j-1,j] > 0 and B in daughters_list:
                        PROB = probas[A][B]*table[B][j-1,j]
                        if PROB > table[A][j-1,j]:
                            table[A][j-1,j] = PROB
                            back[A][j-1,j] = b
                            added = True
                            
                    
        for i in range(j-2,-1,-1):
            for k in range(i+1,j):
                #------- for all {A|A->BC in grammar} ------
                for a, A in enumerate(grammar['parents']):
                    for bc, BC in enumerate(grammar['daughters'][A]):
                        if len(BC.split(' '))==2: # Check that BC is actually not B 
                            B = BC.split(' ')[0]
                            b = grammar['parents'].index(B)
                            C = BC.split(' ')[1]
                            c = grammar['parents'].index(C)
                            if table[B][i,k]>0 and table[C][k,j]>0: #-- and table[i,k,B]>0 and table[k,j,C]>0---
                                #------- for all {A|A->BC in grammar}--------
                                PROB = probas[A][BC]*table[B][i,k]*table[C][k,j]
                                if table[A][i,j]<PROB:
                                    table[A][i,j] = PROB
                                    back[A][i,j] = [k,b,c]
                                    
                    added = True
                    daughters_list = grammar['daughters'][A]
                    while added:
                        added = False
                        for b,B in enumerate(grammar['parents']):
                            if table[B][i,j] > 0 and B in daughters_list:
                                PROB = probas[A][B]*table[B][i,j]
                                if PROB > table[A][i,j]:
                                    table[A][i,j] = PROB
                                    back[A][i,j] = b
                                    added = True
                        
    
    return back, table




def build_tree(words,grammar,back,table,start,end,index,S=''):
    
    indexes = back[start,end,index]
#     print(indexes)
    if any(indexes==[0,0,0]):
        return words[start]
    else:
        [k,b,c] = indexes.astype('int')
        S += "(" +grammar['parents'][b]+" "+ build_tree(words,grammar,back,table,start,k,b) +")"
        S += "(" +grammar['parents'][c]+" "+ build_tree(words,grammar,back,table,k,end,c) +")"
        return S
        
#Version which handles unary rules
def build_tree_u(words,grammar,back,table,start,end,index,S=''):
     
    indexes = back[grammar['parents'][index]][start,end] 
#     print(indexes)
    if indexes=={}:
        return words[start]
    else:
        if type(indexes) is int:
            S += "(" +grammar['parents'][indexes]+" "+ build_tree_u(words,grammar,back,table,start,end,indexes) +")"
        else:
            [k,b,c] = indexes
            S += "(" +grammar['parents'][b]+" "+ build_tree_u(words,grammar,back,table,start,k,b) +")"
            S += "(" +grammar['parents'][c]+" "+ build_tree_u(words,grammar,back,table,k,end,c) +")"
        return S

def to_words(norm_sentence):
    norm_sentence = norm_sentence[7:]
    words = []
    idx = norm_sentence.find(' (')
    while idx!=-1:
        idx_end_type = norm_sentence.find(' ',idx+1)
        idx_end_word = norm_sentence.find(')',idx_end_type)
        word = norm_sentence[idx_end_type+1:idx_end_word]
        if word.find(' ')!=-1:
            word = word.split(' ')[-1] 
        if word.find("'")!=-1:
            words.append('"'+word+'"')
        else:
            words.append("'"+word+"'")
        idx = norm_sentence.find(' (',idx_end_word)
        
    return words

def stdin_to_words(sentence):
    
    words = []
    sent = sentence.split(' ')
    for i,word in enumerate(sent):
        if word.find("'")!=-1:
            words.append('"'+word+'"')
        else:
            words.append("'"+word+"'")
            
    return words

def getTypes(corpus):
    types = []
    idx = 0
    idx = corpus.find('(',idx)
    while idx!=-1:
        idx_end = corpus.find(' ',idx)
        t = corpus[idx+1:idx_end]
        if t not in types:
            types.append(t)
            
        idx = corpus.find('(',idx_end)

    type_word = set()
    for t in types:
        if len(t)>0:
            c = t.find('-')
            type_word.add(t[:c] if c !=-1 else t)
            
    type_word = list(type_word)
    return type_word
    
def trimm_derived_rules(corpus, type_word):
    corpus_out = corpus
    for t in type_word:
        idx = corpus_out.find('('+t+'-',0)
        while idx!=-1:
            idx_end = corpus_out.find(' ',idx)
            if idx_end-idx > len(t)+1 and corpus_out[idx+1:idx_end] not in type_word:
                corpus_out = corpus_out.replace(corpus_out[idx:idx_end],'('+t)
                
            idx = corpus_out.find('('+t+'-',idx_end)
            
        idx = corpus_out.find("(NP::")
        while idx!=-1:
            idx_end = corpus_out.find(' ',idx)
            if idx_end-idx > len(t)+1 and corpus_out[idx+1:idx_end] not in type_word:
                
                corpus_out = corpus_out.replace(corpus_out[idx:idx_end],"(NP")
    
            idx = corpus_out.find("(NP::",idx_end)
            
    return corpus_out
    
def main():
    
    import sys
    print(sys.executable)
    
    print('Cleaning the corpus')
    with open(TREEPATH,'r') as f:
        corpus = f.read()    
    
    sentences = corpus.split('( (SENT (')[1:]
    
    # ### Get all possible word types in the corpus
    
    type_word = getTypes(corpus)
    
# ### Remove all types with a dash, and write it in a txt file

    corpus_out = trimm_derived_rules(corpus, type_word)
        
    with open('corpus_out.txt','w') as f:
        f.write(corpus_out)
    
    with open('corpus_out.txt','r') as f:
        text = f.readlines()
    
    # ### Create splits
    
    print('Reading corpus')
    
    prop_train = 0.8
    prop_test = 0.1
    prop_dev = 0.1
    
    #rand_idx = np.random.permutation(len(text))
    rand_idx = range(len(text))
    idx_end_train = int(prop_train*len(text))
    idx_start_test = int((1-prop_test)*len(text))
    
    corpus_train = [text[t] for t in rand_idx[:idx_end_train]]
    corpus_dev = [text[t] for t in rand_idx[idx_end_train:idx_start_test]]
    corpus_test = [text[t] for t in rand_idx[idx_start_test:]]
    
    
    print('Training corpus '+'('+str(prop_train*100)+'%) :' + str(len(corpus_train)))
    print('Development corpus '+'('+str(prop_dev*100)+'%) :' + str(len(corpus_dev)))
    print('Test corpus '+'('+str(prop_test*100)+'%) :' + str(len(corpus_test)))
    

    print('Converting all sentences to Chomsky Normal Format')
    
    print('Training set')
    corpus_train_rules = convert_to_CNF(corpus_train)
    print('Development set')
    corpus_dev_rules = convert_to_CNF(corpus_dev)
    print('Testing set')
    corpus_test_rules = convert_to_CNF(corpus_test)
    
    print('Done')    
    
    print('Extracting the PCFG from the corpus')    
    (grammar_parents, grammar_daughters, P) = extractPCFGu(corpus_train_rules,corpus_dev_rules,corpus_test_rules)

    print('Found '+str(sum([len(grammar_daughters[s]) for s in list(grammar_daughters.keys())]))+' rules')
    print('length grammar_daughter '+str(len(grammar_daughters)))
    grammar = {'parents':grammar_parents,'daughters':grammar_daughters}
    
    import nltk.draw as dr

    print('Please write the text you would like to parse (one space between each word, once sentence per line)')
    print('Press ENTER twice to end')
    
    sentences = []
    while True:
        sentence = input()
        
        if sentence:
            sentences.append(sentence)            
        else:
            break
    
    print('Found '+str(len(sentences))+' sentences.')
    print('Now processing...')
    
    for i,sentence in enumerate(sentences):
        print('sentence '+str(1+i)+': '+sentence)
        words = stdin_to_words(sentence)
        fail = False
        for word in words:
            if corpus.find(' '+word[1:-1]+')')==-1:
                print(word[1:-1]+' not found in corpus, algorithm will fail, moving on to the next sentence.')
                fail = True
        while fail is False:
            print('Computing Probabilistic CKY table')
            b,t = P_CKY_u(words,grammar,P)
            idx_max = 1
            print('Done')
            
            print('Building the most probable tree')
            tree = build_tree_u(words,grammar,b, t, 0,-1,idx_max)
            tr = Tree.fromstring('( ( SENT'+ tree + '))')
            
            print('Drawing the tree: Please close the window to proceed to next sentence')
            dr.draw_trees(tr)
            
            fail = True
        
    print('Finished')
    
main()