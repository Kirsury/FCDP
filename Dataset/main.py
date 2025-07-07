import os
import io
import json
from git import Repo
# from clang.cindex import Index, CursorKind, Config


# ======= 启动分析 =======
if __name__ == "__main__":
    # 设置你本地的项目路径
    repo_path = "/path/to/your/git-c-project"

    # 控制台调试建议
    # results = process_repo(repo_path)
    repo = Repo(repo_path)
    assert not repo.bare
    result = []
    dts_keyword = "DTS"

    for branch in repo.branches:
        print(f"Checking branch: {branch.name}")
        repo.git.checkout(branch)

        for commit in repo.iter_commits(branch):
            is_dts = dts_keyword in commit.message
            parent = commit.parents[0] if commit.parents else None
            diffs = commit.diff(parent, create_patch=True)

            for diff in diffs:
                if not diff.b_path or not diff.b_path.endswith('.c'):
                    continue

                print(f"Processing file: {diff.b_path} in commit {commit.hexsha}")

                try:
                    if diff.a_blob:
                        old_code = diff.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                    if diff.b_blob:
                        new_code = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                except Exception as e:
                    print(f"Error reading blobs: {e}")
                    continue

                #
                # # 函数提取
                # old_funcs = extract_functions_with_ast(old_code, "_old_tmp.c")
                # new_funcs = extract_functions_with_ast(new_code, "_new_tmp.c")
                #
                # if is_dts:
                #     # 修改前的所有函数
                #     for name, data in old_funcs.items():
                #         label = 1 if name in new_funcs and function_changed(data, new_funcs[name]) else 0
                #         result.append({
                #             'branch': branch.name,
                #             'dir': diff.a_path,
                #             'func_name': name,
                #             'label': label,
                #             'commit': commit.hexsha,
                #             'ast': str(data['ast'].displayname),
                #         })
                #
                #     # 修改后的函数统一标记为0
                #     for name, data in new_funcs.items():
                #         result.append({
                #             'branch': branch.name,
                #             'dir': diff.b_path,
                #             'func_name': name,
                #             'label': 0,
                #             'commit': commit.hexsha,
                #             'ast': str(data['ast'].displayname),
                #         })
