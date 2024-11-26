import numpy as np
import sys

def delete_symbols(sentences_list_initial):
    symbols = [",", ".", ":", ";", "!", "?"]
    sentences_list = []
    for sentence in sentences_list_initial:
        for i in sentence:
            if i in symbols:
                sentence = sentence.replace(i, "")
        sentences_list.append(sentence)
    return sentences_list

def create_terms_list(terms_list_nested):
    terms_list = []
    for row in terms_list_nested:
        terms_list.extend(row)
    return list(set(terms_list))

def split_to_terms(sentences_list_clean):
    terms = []
    for sentence in sentences_list_clean:
        term = sentence.split()
        terms.append(term)
    return terms

def process_input():
    input_data = sys.stdin.read().strip().lower().split('\n')

    num_sentences = int(input_data[0])
    #kolumny macierzy C
    sentences_clean = delete_symbols(input_data[1:-2])
    # wiersze macierzy C
    terms_list = sorted(create_terms_list(split_to_terms(sentences_clean)))
    query_list = input_data[-2].split()
    k = int(input_data[-1])

    return query_list, terms_list, sentences_clean, k, num_sentences
def create_matrix_C(sentences_clean, terms_list, num_sentences):
    x= len(terms_list)
    C = np.zeros((x, num_sentences), dtype=int)
    for i in range(num_sentences):
        for j in range(len(terms_list)):
            if terms_list[j] in sentences_clean[i]:
                C[j][i] = 1
            else:
                C[j][i] = 0
    return C

def create_q(terms_list, query_list):
    q = np.zeros(len(terms_list))
    for i in range(len(terms_list)):
        if terms_list[i] in query_list:
            q[i] = 1
        else:
            q[i] = 0
    return q

def calculate_Ck_qk(C, q, k):
    #dekompozycja macierzy
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    S = np.diag(s)

    #obliczanie Ck
    sk = np.take(s, range(k), axis=0)
    Sk = np.diag(sk)
    VkT = np.take(Vt, range(k), axis=0)
    Ck = Sk.dot(VkT)

    #obliczanie qk
    Sk_odw = np.linalg.inv(Sk)
    UkT = np.take(U.T, range(k), axis=0)
    qk = Sk_odw.dot(UkT).dot(q)

    return Ck, qk

def calculate_cosinus(Ck, qk):
    results=[]
    for document in Ck.T:
        cosinus = np.dot(document, qk) / (np.linalg.norm(document) * np.linalg.norm(qk))
        results.append((float(cosinus)))
        results_formatted = list(float(x) for x in(np.round(results, decimals=2)))
    return results_formatted

if __name__ == '__main__':
    query_list, terms_list, sentences_clean, k, num_sentences = process_input()
    C = create_matrix_C(sentences_clean, terms_list, num_sentences)
    q = create_q(terms_list, query_list)
    Ck,qk = calculate_Ck_qk(C, q, k)
    print(calculate_cosinus(Ck, qk))

