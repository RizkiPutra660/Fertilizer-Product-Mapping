import csv
import math
import numpy as np
import pandas as pd
import random
import re
import sys
import time

from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer



def re_partition_sub(all_re, part_re, replacer, p_word, debug_mode=False):
    re_formula = re.search(all_re, p_word)
    if (re_formula):
        res = re.findall(all_re, p_word)
        for r in res:
            p_word = p_word.replace(r, re.sub(part_re, replacer, r))

    return p_word


def re_partition_ins(all_re, insertion, idx, p_word):
    re_formula = re.search(all_re, p_word)
    if (re_formula):
        res = re.findall(all_re, p_word)
        for r in res:
            p_word = p_word.replace(r, (r[:idx]+insertion+r[idx:]))
            
    return p_word


def get_clean_string(p_word, debug_mode=False):
    local_replacer = {
        "\\xa": " ",
        "ù": "u",
        "°": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
    }

    for (k, v) in local_replacer.items():
        p_word = p_word.replace(k, v)
    if debug_mode: print(f"Penemuan simbol-simbol mirip                              : {p_word}")
        
    p_word = re_partition_sub(r"[^0-9]1/4[^0-9]", r"1/4", "¼", p_word)
    p_word = re_partition_sub(r"[^0-9]1/2[^0-9]", r"1/2", "½", p_word)
    p_word = re_partition_sub(r"[^0-9]3/4[^0-9]", r"3/4", "¾", p_word)
    if debug_mode: print(f"Penemuan simbol pecahan                                   : {p_word}")

    p_word = p_word.lower()
    if debug_mode: print(f".lower() semua huruf                                      : {p_word}")

    p_word = re.sub(r"[^0-9a-z¼½¾]+", " ", p_word)
    if debug_mode: print(f"Replace non angka atau huruf                              : {p_word}")

    p_word = re_partition_ins(r"[a-z][0-9¼½¾]", " ", 1, p_word)
    p_word = re_partition_ins(r"[0-9¼½¾][a-z]", " ", 1, p_word)
    if debug_mode: print(f"Pisahkan char angka yg menempel char huruf dan sebaliknya : {p_word}")
    
    p_word_splitter = p_word.split()
    p_word_splitter.sort()
    p_word = " ".join(p_word_splitter)
    if debug_mode: print(f"Urutkan kata atau bilangan yg sudah terpisah spasi        : {p_word}\n")

    return p_word


def get_cosine_similarity_score(vec1, vec2):
    # ASUMSI PANJANG SELALU SAMA
    norm = math.sqrt(np.dot(vec1, vec1) * np.dot(vec2, vec2))
    similarity_score = np.dot(vec1, vec2) / norm
    return similarity_score


'''
def get_similarity_score(vec1, vec2, cmpr='div', calc='quadratic', solve_both_zero=True, debug_mode=False):
    l_cmpr = ["and", "delta", "div"]
    l_calc = ["arithmetic", "cosine", "geometric", "harmonic", "quadratic"]
    vec_result = []
    similarity_score = 0.0

    if (len(vec1) == len(vec2)) and (cmpr in l_cmpr) and (calc in l_calc):
        vec_result = [
            int(solve_both_zero) if ((v1 == 0) and (v2 == 0) and cmpr in ["and", "div"])
            else int((v1 == 0) == (v2 == 0)) if (cmpr == "and")
            else round(min(v1,v2)/max(v1,v2), 4) if (cmpr == "div")
            else abs(v1-v2) if (cmpr == "delta")
            else 0.0 for (v1, v2) in zip(vec1, vec2)
        ] if (calc != "cosine") else []

        if (cmpr in ["geometric", "harmonic"]):
            vec_result = [v for v in vec_result if (v != 0)]
            if (cmpr == "harmonic"):
                vec_result = [(1/v) for v in vec_result]
        
        similarity_score = (
            (np.sum(vec_result) / len(vec_result)) if (cmpr == "arithmetic")
            else get_cosine_similarity_score(vec1, vec2) if (cmpr == "cosine")
            else (math.pow(np.prod(vec_result), len(vec_result))) if (cmpr == "geometric")
            else (len(vec_result) / np.sum(vec_result)) if (cmpr == "harmonic")
            else (math.sqrt(np.dot(vec_result, vec_result) / len(vec_result)))
        )

    similarity_score = round(similarity_score, 6)
    if debug_mode:
        print(f"Kombinasi perbandingan vector dan perhitungan similarity : {formula.upper()}")
        if (l_calc in ["and", "div"]):
            print(f"Handle elemen nol (0 vs 0)      : {str(solve_both_zero)}")
        print(list(vec1))
        print(list(vec2))
        print(vec_result)
        print(similarity_score)

    return similarity_score
'''


def get_similarity_score(vec1, vec2, formula='div', solve_both_zero=True, debug_mode=False):
    formulas = ["and", "cosine", "div", "pivot"]
    vec_result = []
    similarity_score = 0.0

    if (formula in formulas) and (len(vec1) == len(vec2)):
        if (formula == "pivot") and (np.sum(vec2) > np.sum(vec1)):
            vec_temp = vec1
            vec1 = vec2
            vec2 = vec_temp
        
        vec_result = [
            int(solve_both_zero) if ((v1 == 0) and (v2 == 0))
            else int((v1 == 0) == (v2 == 0)) if (formula == "and")
            else round(min(v1,v2)/max(v1,v2), 4) if (formula == "div")
            else min(1.0, round(v1/v2, 4)) if ((v1 > 0) and (v2 > 0) and (formula == "pivot"))
            else 0.0 for (v1, v2) in zip(vec1, vec2)
        ] if (formula != "cosine") else []

        similarity_score = (
            get_cosine_similarity_score(vec1, vec2) if (formula == "cosine")
            else round(sum(vec_result)/len(vec_result), 6)
        )

    if debug_mode:
        print(f"\nRumus similarity yang digunakan : {formula.upper()}")
        if (formula not in ["cosine", "distance"]):
            print(f"Handle elemen nol (0 vs 0)      : {str(solve_both_zero)}")
        print(list(vec1))
        print(list(vec2))
        print(vec_result)
        print(similarity_score)

    return similarity_score


def get_my_string_vectorizer(s1, s2, debug_mode=False):
    cleaned_s1 = get_clean_string(s1, debug_mode)
    cleaned_s2 = get_clean_string(s2, debug_mode)
    
    words = set([w for w in cleaned_s1.split()] + [w for w in cleaned_s2.split()])
    sublen_words = [len(w) for w in words]
    max_sublen = (max(sublen_words) if (len(sublen_words) > 0) else 0)
    charss = set([c[i:i+j] for c in words for i in range(len(c)) for j in range(1, 4)])
    vocabs = sorted(list(set().union(words, charss, [" "])))

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, max_sublen), vocabulary=vocabs)
    weights = vectorizer.fit_transform([cleaned_s1, cleaned_s2]).toarray()

    if debug_mode:
        features = list(vectorizer.get_feature_names_out())
        debug_df = pd.DataFrame(weights, columns=features)
        debug_df = debug_df[vocabs].T
        debug_df.columns = [s1, s2]
        print(debug_df.to_string())
        print(f"\nString-1 (cleaned)       : \"{cleaned_s1}\"")
        print(f"String-2 (cleaned)       : \"{cleaned_s2}\"")

    return weights


def get_string_similarity_score(s1, s2, formula='div', solve_both_zero=True, debug_mode=False):
    start_dt = time.time()
    vectorizer_arr = get_my_string_vectorizer(s1, s2, debug_mode)
    similarity_score = get_similarity_score(
        vectorizer_arr[0], vectorizer_arr[1],
        formula, solve_both_zero, debug_mode
    )

    end_dt = time.time()
    delta = int(1000 * round(end_dt - start_dt, 3))
    if (debug_mode):
        print(f"\nTingkat kemiripan        : {round(100*similarity_score, 2)}%")
        print(f"Waktu eksekusi penilaian : {delta} ms\n")

    return similarity_score


def get_my_predict(
    p_string, targets, min_threshold=0.0001, 
    formula='div', solve_both_zero=True, debug_mode=False
):
    start_dt = time.time()
    val = np.vectorize(get_string_similarity_score)(p_string, targets, formula, solve_both_zero)
    results = max(zip(targets, val), key=lambda x: x[1])
    condition = (results[1] > 0) and (results[1] >= min_threshold)
    closest_str = (results[0] if condition else "")
    closest_score = (results[1] if condition else 0)

    end_dt = time.time()
    delta = int(1000 * round(end_dt - start_dt, 3))
    if debug_mode:
        condition = (results[1] > 0)
        debug_str = (results[0] if condition else "")
        debug_score = (
            get_string_similarity_score(
                p_string, debug_str, formula, solve_both_zero, debug_mode
            ) if condition else 0
        )
        print(f"\nWaktu eksekusi pencarian : {delta} ms")
        if (results[1] < min_threshold):
            print(f"Target tidak ditemukan! Tingkat kemiripan belum memenuhi batas minimal, perlu lebih dari atau sama dengan {round(100*min_threshold)}%")

    return (closest_str, closest_score)


def extract_batch_predict(
    datasets,
    targets,
    min_threshold=0.0001,
    formula='div',
    solve_both_zero=True,
    batch_size=557,
    filename='RESULTS SIMILARITY VERSION/transactions_prediction',
    first_file_id=0,
    last_file_id=sys.maxsize,
    sample_split=False,
    rand_state=0,
    debug_mode=False
):
    min_file_id = 0
    max_file_id = math.ceil(datasets.shape[0]/batch_size)
    max_row_id = datasets.shape[0]
    
    local_first_file_id = (min(first_file_id, max_file_id) if (first_file_id > 0) else min_file_id)
    local_last_file_id = (min(last_file_id, max_file_id) if (last_file_id > 0) else max_file_id)
    file_count = local_last_file_id - local_first_file_id

    first_row_id = min(local_first_file_id * batch_size, max_row_id)
    last_row_id = min(first_row_id + file_count*batch_size, max_row_id)
    delta_row_id = last_row_id - first_row_id
    row_count = min(delta_row_id, max_row_id)
    local_batch_size = (min(batch_size, row_count) if (batch_size > 0) else row_count)
    
    if debug_mode:
        print(f"{{min_file_id}}: {min_file_id}")
        print(f"{{first_file_id}}: {local_first_file_id}")
        print(f"{{last_file_id}}: {local_last_file_id}")
        print(f"{{max_file_id}}: {max_file_id}")
        print(f"{{file_count}}: {file_count}")
        print()
        print(f"{{first_row_id}}: {first_row_id}")
        print(f"{{last_row_id}}: {last_row_id}")
        print(f"{{max_row_id}}: {max_row_id}")
        print()

    print(f"\nJob executing {row_count} rows started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
    remark = ""
    
    if sample_split:
        local_df = datasets.sample(n=last_row_id, random_state=rand_state)
        remark = f"{remark}randstate{rand_state:03}_"
    else:
        local_df = datasets

    fid = local_first_file_id
    rid = first_row_id
    while (fid < local_last_file_id):
        next_fid = fid + 1
        next_rid = min(rid + local_batch_size, max_row_id)
        local_filename=f"{filename}_{remark}{local_batch_size}rows_{(next_fid):04}.csv"
        
        if (not debug_mode):
            df_batch = local_df.iloc[rid:next_rid].copy()
            df_batch['predict_product_name_to_sku'] = df_batch['Product Name'].apply(lambda x: get_my_predict(
                x, targets['Product SKU'], min_threshold, formula, solve_both_zero
            ))
            df_batch[['closest_product_sku', 'similarity_to_closest_product_sku']] = pd.DataFrame(
                list(df_batch['predict_product_name_to_sku']), index=df_batch.index
            )
            df_batch.drop(columns=['predict_product_name_to_sku'])
            #selected_cols = ['id', 'Product Name', 'closest_product_sku', 'similarity_to_closest_product_sku']
            df_batch_joined = pd.merge(
                df_batch.reset_index(names='id'), targets, how='left',
                left_on='closest_product_sku', right_on='Product SKU', sort=False
            )
            df_batch_joined.to_csv(local_filename, sep=";", index=False, quoting=csv.QUOTE_NONNUMERIC)

        print(f"File [{local_filename}] for rowid[{rid}:{next_rid}]\ngenerated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
        fid, rid = next_fid, next_rid

    print(f"\nJob succeeded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
    #res = ([] if debug_mode else df_batch_joined)
    #return res