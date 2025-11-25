import sys
import datetime
import git
from typing import Optional, Tuple, Dict
from tree_sitter import Language, Parser, Node
import tree_sitter_c


class CFunctionLocator:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        try:
            self.repo = git.Repo(repo_path)
        except git.exc.InvalidGitRepositoryError:
            raise ValueError(f"Error: {repo_path} is not a valid git repository.")
        except git.exc.NoSuchPathError:
            raise ValueError(f"Error: Path {repo_path} does not exist.")

        # 初始化 Tree-sitter
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser(self.language)

    def get_commit_by_time(self, submit_time: datetime.datetime) -> git.Commit:
        """根据提交时间找到最近的一次提交"""
        target_timestamp = submit_time.timestamp()

        # 简单遍历（实际生产环境可优化为二分查找或 git log -n 1 --before）
        best_commit = None
        for commit in self.repo.iter_commits():
            if commit.committed_date <= target_timestamp:
                best_commit = commit
                break

        if not best_commit:
            # 如果没找到更早的，使用最早的一次提交
            best_commit = list(self.repo.iter_commits())[-1]

        return best_commit

    def get_file_content_at_commit(self, commit: git.Commit, file_path: str) -> bytes:
        """获取指定 Commit 时刻的文件内容"""
        try:
            target_file = commit.tree / file_path
            return target_file.data_stream.read()
        except KeyError:
            return b""

    def locate_function_node(self, source_code: bytes, line_number: int) -> Optional[Node]:
        """
        根据行号定位 function_definition 节点
        """
        tree = self.parser.parse(source_code)
        root_node = tree.root_node

        target_row = line_number - 1
        if target_row < 0:
            return None

        # 找到该行的节点
        node = root_node.descendant_for_point_range((target_row, 0), (target_row, 1))

        # 向上回溯直到找到 function_definition
        while node:
            if node.type == 'function_definition':
                return node
            node = node.parent
        return None

    def extract_function_info(self, source_code: bytes, line_number: int) -> Dict[str, str]:
        """
        核心方法：提取函数名和完整代码
        """
        func_node = self.locate_function_node(source_code, line_number)

        if not func_node:
            return {"name": None, "code": None}

        # 1. 提取函数名
        func_name = self._extract_function_name_from_node(func_node)

        # 2. 提取完整函数代码
        # Tree-sitter 的 text 属性返回的是 bytes，需要解码
        # 或者使用 start_byte / end_byte 从源代码切片（更推荐，保留原始格式）
        func_code_bytes = source_code[func_node.start_byte: func_node.end_byte]
        func_code = func_code_bytes.decode('utf-8', errors='replace')

        return {
            "name": func_name,
            "code": func_code
        }

    def _extract_function_name_from_node(self, func_def_node: Node) -> str:
        """从 function_definition 节点中提取函数名 (处理指针、嵌套等情况)"""
        declarator = func_def_node.child_by_field_name('declarator')
        if not declarator:
            return "Unknown"

        curr = declarator
        while curr:
            if curr.type == 'function_declarator':
                next_decl = curr.child_by_field_name('declarator')
                if next_decl:
                    curr = next_decl
                else:
                    break
            elif curr.type in ('pointer_declarator', 'parenthesized_declarator'):
                curr = curr.child_by_field_name('declarator')
            elif curr.type == 'identifier':
                return curr.text.decode('utf-8')
            else:
                break

        # Fallback
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


# ==========================================
# 测试入口
# ==========================================

if __name__ == "__main__":
    # 使用之前测试通过的 cJSON 数据
    # 为了演示获取完整函数，我们使用 155 行 (位于 cJSON_New_Item 内部)

    REPO_PATH = "./cJSON_repo"  # 确保此目录存在且是git仓库
    FILE_REL_PATH = "cJSON.c"
    ERROR_LINE = 152  # case_insensitive_strcmp
    # https://github.com/DaveGamble/cJSON/blob/87d8f09/cJSON.c#L150
    ERROR_TIME_STR = "2023-12-30 10:00:00"

    print(f"Searching in {REPO_PATH} | {FILE_REL_PATH}:{ERROR_LINE} @ {ERROR_TIME_STR} ...")

    output = get_function_context(REPO_PATH, FILE_REL_PATH, ERROR_LINE, ERROR_TIME_STR)

    if "error" in output:
        print(f"❌ Failed: {output['error']}")
    else:
        print(f"\n✅ Found Function Name: {output['name']}")
        print(f"📍 Commit: {output['commit_hex'][:7]} ({output['commit_time']})")

        if output['code']:
            print("-" * 30)
            print("📜 Full Function Source Code:")
            print("-" * 30)
            print(output['code'])
            print("-" * 30)
        else:
            print("❌ Line is not inside a function definition.")