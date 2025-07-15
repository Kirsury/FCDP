import os

current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, 'raw')
file_paths = {os.path.join(target_path, f):f for f in os.listdir(target_path)}

import json

for fp in file_paths.keys():
    out = []
    is_defect = set()
    no_defect = set()
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for line in data:
        if line['label']==1 and line['code'] not in is_defect:
            is_defect.add(line['code'])
    for line in data:
        if line['label']==0 and line['code'] not in no_defect and line['code'] not in is_defect:
            no_defect.add(line['code'])
    for line in data:
        if line['label'] == 1 and line['code'] in is_defect:
            out.append(line)
        if line['label'] == 0 and line['code'] in no_defect:
            out.append(line)
    with open(os.path.join(current_dir, 'cleaned', file_paths[fp]), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(len(out), '/', len(data))



