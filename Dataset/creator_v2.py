import hashlib
import re
import os
import json
from git import Repo
from tqdm import tqdm

from Dataset.CFunctionExtractor import CFunctionExtractor


"""
***This is entrypoint for creating dataset.***
Functions unchanged are defined as no bug.
"""

def extract_dts_id(message):
    match = re.search(r'DTS(\d+)', message)
    return f'DTS{match.group(1)}' if match else 'None'

def extract_and_store(container, file_path, label, branch_name, commit_id, dts_id, diff, diff_path):
    try:
        functions = extractor.extract_functions(file_path)
        for func in functions:
            container.append({
                'repo': repo_name,
                'name': func['name'],
                'signature': func['signature'],
                'code': func['code'],
                'ast': func['ast'],  # 这里假设extractor能返回AST
                'label': label,
                'branch': branch_name,
                'commit': commit_id,
                'dts_id': dts_id,
                'diff': diff,
                'diff_path': diff_path
            })
    except Exception as e:
        print(f"[Error] Failed to extract from {file_path}: {e}")

def ast_hash(ast_dict):
    ast_json = json.dumps(ast_dict, sort_keys=True)  # 结构序列化，确保 key 顺序一致
    return hashlib.sha256(ast_json.encode('utf-8')).hexdigest()

def hash_json_str(ast_json_str):
    return hashlib.sha256(ast_json_str.encode('utf-8')).hexdigest()

if __name__ == '__main__':

    dts_keyword = "DTS"
    extractor = CFunctionExtractor()
    repos_dir = "/dir/to/your/git-projects"
    repos_names = [fn for fn in os.listdir(repos_dir)]
    a_funcs_container = []
    b_funcs_container = []

    for repo_name in repos_names:
        repo_path = os.path.join(repos_dir, repo_name)
        repo = Repo(repo_path)
        for branch in repo.branches:
            print(f"Checking branch: {branch.name}")
            repo.git.checkout(branch)

            for commit in tqdm(list(repo.iter_commits(branch))):
                is_dts = dts_keyword in commit.message
                dts_id = extract_dts_id(commit.message)
                a_label = 1 if is_dts else 0
                parent = commit.parents[0] if commit.parents else None
                diffs = commit.diff(parent, create_patch=True)

                for diff in diffs:
                    if (not diff.a_path or not diff.a_path.endswith('.c')) and (not diff.b_path or not diff.b_path.endswith('.c')):
                        continue

                    if not diff.b_path or not diff.b_path.endswith('.c'):
                        if diff.a_path.endswith('.c'):
                            a_file_path = os.path.join(repo_path, diff.a_path)

                            extract_and_store(a_funcs_container, a_file_path, label=a_label, branch_name=branch.name,
                                              commit_id=commit.hexsha, dts_id=dts_id, diff='a', diff_path=a_file_path)
                            continue

                    if not diff.a_path or not diff.a_path.endswith('.c'):
                        if diff.b_path.endswith('.c'):
                            b_file_path = os.path.join(repo_path, diff.b_path)
                            extract_and_store(b_funcs_container, b_file_path, label=0, branch_name=branch.name,
                                              commit_id=commit.hexsha, dts_id=dts_id, diff='b', diff_path=b_file_path)
                            continue


                    if diff.a_path.endswith('.c') and diff.b_path.endswith('.c'):
                        a_file_path = os.path.join(repo_path, diff.a_path)
                        b_file_path = os.path.join(repo_path, diff.b_path)

                        try:
                            extract_and_store(a_funcs_container, a_file_path, label=a_label, branch_name=branch.name,
                                              commit_id=commit.hexsha, dts_id=dts_id, diff='a', diff_path=b_file_path)
                            extract_and_store(b_funcs_container, b_file_path, label=0, branch_name=branch.name,
                                              commit_id=commit.hexsha, dts_id=dts_id, diff='b', diff_path=b_file_path)
                        except Exception as e:
                            print(f"[Error] Diff extraction failed: {e}")
                            continue

    # 根据AST判断哪些函数变化了
    a_func_map = {f['ast']: f for f in a_funcs_container}
    b_func_map = {f['ast']: f for f in b_funcs_container}
    a_funcs = set(a_func_map.keys())
    b_funcs = set(b_func_map.keys())

    unchanged_funcs_keys = b_funcs - a_funcs
    result = [b_func_map[key] for key in unchanged_funcs_keys]
    other_result = [a_func_map[key] for key in a_funcs]

    with open("no_defect_dataset.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open("or_defect_dataset.json", "w", encoding="utf-8") as f:
        json.dump(other_result, f, ensure_ascii=False, indent=2)
