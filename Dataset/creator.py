import hashlib
import re
import os
import json
from git import Repo
from tqdm import tqdm

from Dataset.CFunctionExtractor import CFunctionExtractor

def extract_dts_id(message):
    match = re.search(r'DTS(\d+)', message)
    return f'DTS{match.group(1)}' if match else 'None'

def extract_and_store(file_path, label, branch_name, commit_id, dts_id):
    try:
        functions = extractor.extract_functions(file_path)
        for func in functions:
            result.append({
                'code': func['code'],
                'ast': func['ast'],  # 这里假设extractor能返回AST
                'label': label,
                'branch': branch_name,
                'commit': commit_id,
                'dts_id': dts_id
            })
    except Exception as e:
        print(f"[Error] Failed to extract from {file_path}: {e}")

def ast_hash(ast_dict):
    ast_json = json.dumps(ast_dict, sort_keys=True)  # 结构序列化，确保 key 顺序一致
    return hashlib.sha256(ast_json.encode('utf-8')).hexdigest()

if __name__ == '__main__':

    extractor = CFunctionExtractor()
    repo_path = "/path/to/your/git-c-project"
    repo = Repo(repo_path)
    result = []

    for branch in repo.branches:
        print(f"Checking branch: {branch.name}")
        repo.git.checkout(branch)

        for commit in tqdm(repo.iter_commits(branch)):
            is_dts = dts_keyword in commit.message
            dts_id = extract_dts_id(commit.message)
            parent = commit.parents[0] if commit.parents else None
            diffs = commit.diff(parent, create_patch=True)

            for diff in diffs:
                if (not diff.a_path or not diff.a_path.endswith('.c')) and (not diff.b_path or not diff.b_path.endswith('.c')):
                    cnt1 += 1
                    continue

                if not diff.a_path or not diff.a_path.endswith('.c'):
                    if diff.b_path.endswith('.c'):
                        cnt2 += 1
                        # 提取b中所有函数，label=0（无缺陷）
                        b_file_path = os.path.join(repo_path, diff.b_path)
                        extract_and_store(b_file_path, label=0, branch_name=branch.name,
                                          commit_id=commit.hexsha, dts_id=dts_id)
                        continue

                if not diff.b_path or not diff.b_path.endswith('.c'):
                    if diff.a_path.endswith('.c'):
                        cnt3 += 1
                        a_file_path = os.path.join(repo_path, diff.a_path)
                        label = 1 if is_dts else 0
                        extract_and_store(a_file_path, label=label, branch_name=branch.name,
                                          commit_id=commit.hexsha, dts_id=dts_id)
                        continue

                if diff.a_path.endswith('.c') and diff.b_path.endswith('.c'):
                    cnt4 += 1
                    a_file_path = os.path.join(repo_path, diff.a_path)
                    b_file_path = os.path.join(repo_path, diff.b_path)

                    try:
                        a_funcs = extractor.extract_functions(a_file_path)
                        b_funcs = extractor.extract_functions(b_file_path)
                    except Exception as e:
                        print(f"[Error] Diff extraction failed: {e}")
                        continue

                    # 用最粗暴的方式判断哪些函数变化了（根据函数名匹配）
                    a_func_map = {f['name']: f for f in a_funcs}
                    b_func_map = {f['name']: f for f in b_funcs}
                    a_func_names = set(a_func_map.keys())
                    b_func_names = set(b_func_map.keys())

                    changed_funcs = a_func_names & b_func_names  # 名字相同但内容可能不同

                    for name in a_func_names:
                        label = 1 if (is_dts and name in changed_funcs) else 0
                        func = a_func_map[name]
                        result.append({
                            'code': func['code'],
                            'ast': func['ast'],
                            'label': label,
                            'branch': branch.name,
                            'commit': commit.hexsha,
                            'dts_id': dts_id
                        })

                    for name in b_func_names:
                        func = b_func_map[name]
                        result.append({
                            'code': func['code'],
                            'ast': func['ast'],
                            'label': 0,
                            'branch': branch.name,
                            'commit': commit.hexsha,
                            'dts_id': dts_id
                        })

    # 最后可选保存
    with open("function_level_defect_dataset.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
