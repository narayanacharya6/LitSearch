import json
import argparse
from calcs import aggregate_recall, aggregate_ndcg
import pandas as pd
from collections import defaultdict

def run(file_name, top_k):
    evals = []
    unique_query_sets, unique_specificity, unique_quality = set(), set(), set()
    for k in top_k:
        with open(file_name) as f:
            query_sets = defaultdict(list)
            for line in f.readlines():
                query = json.loads(line)
                query['retrieved'] = query['retrieved'][:k]
                unique_query_sets.add(query['query_set'])
                unique_specificity.add(query['specificity'])
                unique_quality.add(query['quality'])
                if query['query_set'] in {"inline_acl", "inline_nonacl"}:
                    query_sets["inline"].append(query)
                    if query["quality"] == 1:
                        query_sets["inline_q1"].append(query)
                    elif query["quality"] == 2:
                        query_sets["inline_q2"].append(query)

                    if query["specificity"] == 0:
                        query_sets["inline_broad"].append(query)
                        query_sets["broad"].append(query)
                    else:
                        query_sets["inline_specific"].append(query)
                        query_sets["specific"].append(query)
                else:
                    query_sets["manual"].append(query)
                    if query["quality"] == 1:
                        query_sets["manual_q1"].append(query)
                    elif query["quality"] == 2:
                        query_sets["manual_q2"].append(query)

                    if query["specificity"] == 0:
                        query_sets["manual_broad"].append(query)
                        query_sets["broad"].append(query)
                    else:
                        query_sets["specific"].append(query)
                        query_sets["manual_specific"].append(query)

            for query_set_name, queries in query_sets.items():
                ndcg_results = aggregate_ndcg(queries)
                recall_results = aggregate_recall(queries)
                evals.append({
                    "query_set_name": query_set_name,
                    "num_queries": len(queries),
                    "k": k,
                    # "mean_ndcg": f"{ndcg_results['mean_ndcg']:.3f}",
                    "mean_recall": f"{recall_results['mean_recall']:.3f}",
                    "micro_avg_recall": f"{recall_results['micro_avg_recall']:.3f}"
                })

    print(unique_query_sets, unique_specificity)
    for k in top_k:
        df = pd.DataFrame(data=evals)
        print(df[df["k"] == k].set_index('query_set_name').T.to_markdown())
        print()

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, required=False, default='results/retrieval/LitSearch.title_abstract.specter2.jsonl')
parser.add_argument("--top_k", type=lambda x: [int(y) for y in x.split(',')], required=False, default='5,20')

args = parser.parse_args()
run(args.file_name, args.top_k)