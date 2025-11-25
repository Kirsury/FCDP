import sys
import datetime
from typing import Optional, Tuple
import git
from tree_sitter import Language, Parser, Node
import tree_sitter_c


class CFunctionLocator:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        try:
            self.repo = git.Repo(repo_path)
        except git.exc.InvalidGitRepositoryError:
            print(f"Error: {repo_path} is not a valid git repository.")
            sys.exit(1)

        # 初始化 Tree-sitter
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser(self.language)

    def get_commit_by_time(self, submit_time: datetime.datetime) -> git.Commit:
        """
        根据提交时间找到在这个时间点之前最近的一次提交。
        """
        # 将 datetime 转换为 unix timestamp
        target_timestamp = submit_time.timestamp()

        # 遍历提交历史（从新到旧）
        # 注意：这里简单遍历 HEAD 分支，实际场景可能需要指定分支
        best_commit = None

        for commit in self.repo.iter_commits():
            if commit.committed_date <= target_timestamp:
                best_commit = commit
                break

        if not best_commit:
            # 如果没找到更早的，就返回第一次提交或抛出错误
            print("Warning: No commit found before the specified time. Using the oldest reachable commit.")
            best_commit = list(self.repo.iter_commits())[-1]

        return best_commit

    def get_file_content_at_commit(self, commit: git.Commit, file_path: str) -> bytes:
        """
        获取指定 Commit 时刻的文件内容
        """
        try:
            # git.Tree / file_path 语法
            # 注意 file_path 应该是相对于仓库根目录的路径
            target_file = commit.tree / file_path
            return target_file.data_stream.read()
        except KeyError:
            print(f"Error: File '{file_path}' not found in commit {commit.hexsha[:7]}.")
            return b""

    def find_function_name(self, source_code: bytes, line_number: int) -> Optional[str]:
        """
        根据行号提取 C 函数名
        line_number: 1-based index (通常报错信息是 1 开始的)
        """
        tree = self.parser.parse(source_code)
        root_node = tree.root_node

        # Tree-sitter 使用 0-based index，所以需要 -1
        target_row = line_number - 1

        # 边界检查
        if target_row < 0:
            return None

        # 找到该行对应的节点
        # 我们查找该行第一个字符位置的节点，直到该行结束
        # descendant_for_point_range((start_row, start_col), (end_row, end_col))
        node = root_node.descendant_for_point_range((target_row, 0), (target_row, 1))

        # 向上遍历父节点，直到找到 function_definition
        while node:
            if node.type == 'function_definition':
                return self._extract_function_name_from_node(node)
            node = node.parent

        return None

    def _extract_function_name_from_node(self, func_def_node: Node) -> str:
        """
        从 function_definition 节点中提取函数名。
        C 语言的声明比较复杂（指针、修饰符等），需要定位到 declarator。
        """
        # C 语法结构通常是:
        # function_definition
        #   type: "int"
        #   declarator: function_declarator
        #     declarator: identifier "main"
        #     parameters: parameter_list
        #   body: compound_statement

        child_by_field = func_def_node.child_by_field_name

        declarator = child_by_field('declarator')
        if not declarator:
            return "Unknown"

        # 处理多层嵌套，例如指针函数 *func() 或 属性修饰
        # 我们不断深入 'declarator' 直到找到 'identifier'
        # 或者直接取 function_declarator 的直接文本（包含参数之前的名字）

        curr = declarator
        while curr:
            if curr.type == 'function_declarator':
                # function_declarator 的 declarator 字段通常是函数名或指针嵌套
                next_decl = curr.child_by_field_name('declarator')
                if next_decl:
                    curr = next_decl
                else:
                    break
            elif curr.type == 'pointer_declarator':
                curr = curr.child_by_field_name('declarator')
            elif curr.type == 'parenthesized_declarator':
                curr = curr.child_by_field_name('declarator')
            elif curr.type == 'identifier':
                return curr.text.decode('utf-8')
            else:
                # 兜底：如果结构太复杂，直接返回当前节点的文本
                break

        # 如果还没找到 identifier，尝试直接在 declarator 及其子节点里找 identifier
        # 这是一个简化的查找逻辑
        for i in range(declarator.child_count):
            child = declarator.children[i]
            if child.type == 'function_declarator':
                return child.child_by_field_name('declarator').text.decode('utf-8')

        # 最简单的回退方案：返回 declarator 的文本（可能会包含 * 等符号）
        return declarator.text.decode('utf-8').split('(')[0].strip()


# ==========================================
# 对外接口函数
# ==========================================

def get_function_context(repo_path: str, file_rel_path: str, error_line: int, error_time_str: str):
    """
    输入：仓库路径、文件相对路径、行号、时间字符串
    输出：包含 func_name 和 full_code 的字典
    """
    # 1. 解析时间
    try:
        submit_time = datetime.datetime.strptime(error_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": "Invalid time format. Use YYYY-MM-DD HH:MM:SS"}

    # 2. 初始化定位器
    try:
        locator = CFunctionLocator(repo_path)
    except ValueError as e:
        return {"error": str(e)}

    # 3. 定位 Commit
    commit = locator.get_commit_by_time(submit_time)

    # 4. 获取文件内容
    content = locator.get_file_content_at_commit(commit, file_rel_path)
    if not content:
        return {"error": f"File '{file_rel_path}' not found in commit {commit.hexsha[:7]}"}

    # 5. 提取信息
    result = locator.extract_function_info(content, error_line)

    # 附加元数据以便调试
    result["commit_hex"] = commit.hexsha
    result["commit_time"] = str(datetime.datetime.fromtimestamp(commit.committed_date))

    return result

# ================= 使用示例 =================

def main():
    import os

    # 1. 准备测试数据
    REPO_URL = "https://github.com/DaveGamble/cJSON.git"
    REPO_PATH = "./cJSON_repo"  # 脚本会自动克隆到这里
    FILE_REL_PATH = "cJSON.c"

    # 模拟数据：2023年底的报错，行号 150
    # 在那个历史版本中，150行应该属于 cJSON_New_Item 函数 // case_insensitive_strcmp
    # https://github.com/DaveGamble/cJSON/blob/87d8f09/cJSON.c#L150
    ERROR_LINE = 154
    ERROR_TIME_STR = "2023-12-30 10:00:00"

    # 2. 自动克隆仓库（如果不存在）
    if not os.path.exists(REPO_PATH):
        print(f"Cloning {REPO_URL} into {REPO_PATH} ...")
        try:
            git.Repo.clone_from(REPO_URL, REPO_PATH)
            print("Cloning finished.")
        except Exception as e:
            print(f"Failed to clone repository: {e}")
            return
    else:
        print(f"Repository already exists at {REPO_PATH}")

    # 3. 执行定位逻辑
    submit_time = datetime.datetime.strptime(ERROR_TIME_STR, "%Y-%m-%d %H:%M:%S")
    locator = CFunctionLocator(REPO_PATH)

    print(f"\n--- Analysis Start ---")
    print(f"Target Time: {submit_time}")
    print(f"Target File: {FILE_REL_PATH}")
    print(f"Target Line: {ERROR_LINE}")

    # 获取 Commit
    commit = locator.get_commit_by_time(submit_time)
    print(f"Located Commit: {commit.hexsha[:7]}")
    print(f"Commit Date : {datetime.datetime.fromtimestamp(commit.committed_date)}")
    print(f"Commit Msg  : {commit.message.strip().splitlines()[0]}")

    # 获取内容
    content = locator.get_file_content_at_commit(commit, FILE_REL_PATH)
    # print(f"Content length: {len(content)} bytes")
    if not content:
        print("Error: Could not read file content.")
        return

    # 查找函数
    func_name = locator.find_function_name(content, ERROR_LINE)

    print(f"--- Analysis Result ---")
    if func_name:
        print(f"✅ Line {ERROR_LINE} belongs to function: '{func_name}'")

        # 验证结果
        expected_func = "cJSON_New_Item"
        if func_name == expected_func:
            print(f"(Test Passed: Matched expected function '{expected_func}')")
        else:
            print(f"(Test Warning: Expected '{expected_func}', got '{func_name}')")
    else:
        print(f"❌ Could not determine function name.")


if __name__ == "__main__":
    main()