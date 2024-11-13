import csv
from collections import defaultdict

# Read b.tsv and store data
b_queries = defaultdict(list)
task_name = 'MultiHiertt'
with open(f'outputs/{task_name}/results.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if not row:
            continue
        query_id = row[0]
        corpus_id = row[1] if len(row) > 1 else ''
        b_queries[query_id].append(corpus_id)

# Process a.csv
output_rows = []
processed_queries = set()

with open('result/filtered_merged_result.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        query_id = row[0]
        corpus_id = row[1] if len(row) > 1 else ''
        if query_id in b_queries:
            if query_id not in processed_queries:
                # Write entries from b.tsv
                for corpus_id_b in b_queries[query_id]:
                    output_rows.append([query_id, corpus_id_b])
                processed_queries.add(query_id)
            # Skip the current line (do not write the old entry from a.csv)
        else:
            # Write the row as is
            output_rows.append(row)

# Write output_rows back to a.csv
with open('result/replaced_merged_result.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)
