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
    funcs_container = []

    for repo_name in repos_names:
        repo_path = os.path.join(repos_dir, repo_name)
        repo = Repo(repo_path)

        # 遍历所有分支的最新版本
        for branch in repo.branches:
            print(f"Checking branch: {branch.name}")
            repo.git.checkout(branch)

            head_commit = repo.head.commit

            # 遍历工作区所有 .c 文件
            for root, _, files in os.walk(repo_path):
                for f in files:
                    if f.endswith(".c"):
                        file_path = os.path.join(root, f)
                        try:
                            extract_and_store(
                                funcs_container, file_path,
                                branch_name=branch.name,
                                commit_id=head_commit.hexsha,
                                diff='latest',
                                diff_path=file_path
                            )
                        except Exception as e:
                            print(f"[Error] Extraction failed in {file_path}: {e}")
                            continue


    with open("no_defect_dataset.json", "w", encoding="utf-8") as f:
        json.dump(funcs_container, f, ensure_ascii=False, indent=2)
