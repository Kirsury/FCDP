import os
import io
import json
from git import Repo
# from clang.cindex import Index, CursorKind, Config
from tqdm import tqdm

from Dataset.CFunctionExtractor import CFunctionExtractor

# ======= 方法工具 =======
extractor = CFunctionExtractor() # 提取c文件中的函数方法是extractor.extract_functions(file_path=)

# ======= 启动分析 =======
if __name__ == "__main__":
    """
    构建函数级的缺陷预测数据集
    每条数据的列包括函数code、相应的ast、是否包含缺陷、branch name、commit id、DTS id（如果有缺陷，写入id；如果无缺陷写入'None'）
    """


    # 设置你本地的项目路径
    repo_path = "/path/to/your/git-c-project"

    # 控制台调试建议
    # results = process_repo(repo_path)
    repo = Repo(repo_path)
    assert not repo.bare
    result = []
    dts_keyword = "DTS"

    cnt1=0
    cnt2=0
    cnt3=0
    cnt4=0

    for branch in repo.branches:
        print(f"Checking branch: {branch.name}")
        repo.git.checkout(branch)

        for commit in tqdm(repo.iter_commits(branch)):
            is_dts = dts_keyword in commit.message # true表示有缺陷，反之无缺陷
            """TODO
            提取DTS id
            DTS_id = "DTS"后跟随若干个数字
            """
            parent = commit.parents[0] if commit.parents else None
            diffs = commit.diff(parent, create_patch=True)

            for diff in diffs:
                if (not diff.a_path or not diff.a_path.endswith('.c')) and (not diff.b_path or not diff.b_path.endswith('.c')):
                    cnt1+=1
                    """
                    skip, path of a and b are all illegal.
                    """
                    continue

                if not diff.a_path or not diff.a_path.endswith('.c'):
                    if diff.b_path.endswith('.c'):
                        cnt2+=1
                        """
                        TODO
                        path of a is illegal.
                        提取b中的所有函数，这些函数标记为无缺陷的函数，label为1
                        """
                        continue
                if not diff.b_path or not diff.b_path.endswith('.c'):
                    if  diff.a_path.endswith('.c'):
                        cnt3+=1
                        """
                        TODO
                        path of b is illegal.
                        提取a中的所有函数，
                        如果is_dts为真，这些函数被标记为有缺陷的函数，label为1；反之如果为假，这些函数标记为无缺陷，label为0.
                        """
                        continue
                if diff.a_path.endswith('.c') and diff.b_path.endswith('.c'):
                    cnt4+=1
                    """
                    TODO
                    分别提取a和b中的所有函数，a中的函数对应修改前的文件中的函数，b中的函数对应修改后的文件中的函数。
                    比对a和b中的函数可以发现哪些函数是被修改的。
                    如果is_dts为真，那么a中被修改的函数被标记为有缺陷的函数，label为1；a中剩余的未修改的函数和b中的函数标记为无缺陷，label为0.
                    如果is_dts为假，那么所有函数都被标记为无缺陷，label为0.
                    """
                    continue

                print(f"Processing file: {diff.b_path} in commit {commit.hexsha}")

                # try:
                #     if diff.a_blob:
                #         old_code = diff.a_blob.data_stream.read().decode('utf-8', errors='ignore')
                #     if diff.b_blob:
                #         new_code = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                # except Exception as e:
                #     print(f"Error reading blobs: {e}")
                #     continue

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
