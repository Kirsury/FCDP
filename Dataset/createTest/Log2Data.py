import os
from datetime import datetime
from git import Repo, GitCommandError
import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node

# 初始化 Tree-sitter C 语言解析器
C_LANGUAGE = Language(tsc.language())
parser = Parser(C_LANGUAGE)


def get_function_name_from_node(node: Node, source_bytes: bytes):
    """
    从 function_definition 节点中提取函数名。
    C 语言的声明比较复杂（如 static void* func_name(...)），需要递归查找 identifier。
    """
    # 通常结构是: function_definition -> declarator -> function_declarator -> declarator -> identifier
    # 我们通过字段名 'declarator' 深入查找
    curr = node.child_by_field_name('declarator')

    while curr:
        if curr.type == 'identifier':
            return source_bytes[curr.start_byte: curr.end_byte].decode('utf-8')
        # 处理指针函数、带属性的函数等情况，继续找内部的 declarator
        # 例如: *my_func(int a) 的 AST 嵌套
        next_node = curr.child_by_field_name('declarator')
        if not next_node:
            # 如果没有 declarator 字段了，尝试直接找 identifier 子节点
            for child in curr.children:
                if child.type == 'identifier':
                    return source_bytes[child.start_byte: child.end_byte].decode('utf-8')
            break
        curr = next_node

    return "<unknown_function>"


def analyze_code_context(repo_path, file_path, target_time, target_line_number):
    """
    综合分析函数：
    1. Git定位 Commit 和 Diff
    2. Tree-sitter 解析并定位函数
    """
    try:
        repo = Repo(repo_path)

        # 1. 路径处理
        if os.path.isabs(file_path):
            rel_file_path = os.path.relpath(file_path, repo_path)
        else:
            rel_file_path = file_path

        # 2. Git: 查找时间点前的最后一次提交
        commits = list(repo.iter_commits(paths=rel_file_path, until=target_time, max_count=1))
        if not commits:
            return {"error": f"在 {target_time} 之前未找到文件 {rel_file_path} 的提交记录。"}

        target_commit = commits[0]
        commit_hash = target_commit.hexsha

        # 3. Git: 获取 Diff
        diff_content = repo.git.show(commit_hash, "--", rel_file_path)

        # 4. Git: 获取该 Commit 时刻的完整文件内容 (用于 AST 解析)
        # 格式: git show HASH:path/to/file
        try:
            file_content_blob = repo.git.show(f"{commit_hash}:{rel_file_path}")
            file_bytes = file_content_blob.encode('utf-8')  # Tree-sitter 需要 bytes
        except GitCommandError:
            return {"error": "无法获取历史文件内容，文件可能已被删除或重命名。"}

        # 5. Tree-sitter: 解析
        tree = parser.parse(file_bytes)
        root_node = tree.root_node

        # Tree-sitter 是 0-indexed (行号从0开始)，用户输入通常是 1-indexed
        row = target_line_number - 1

        # 检查行号范围
        if row >= len(file_bytes.splitlines()):
            return {"error": f"行号 {target_line_number} 超出文件范围。"}

        # 6. 定位 AST 节点
        # 找到覆盖该行的最小节点
        # point 格式为 (row, column)，我们只关心行，列设为0到行尾
        line_node = root_node.descendant_for_point_range((row, 0), (row, 9999))

        # 7. 向上回溯查找函数定义
        current_node = line_node
        function_name = None

        while current_node:
            if current_node.type == 'function_definition':
                function_name = get_function_name_from_node(current_node, file_bytes)
                break
            current_node = current_node.parent

        if not function_name:
            return {
                "found": False,
                "reason": "指定代码行不属于任何 C 函数 (可能是全局变量、宏或注释)",
                "file": rel_file_path,
                "commit_hash": commit_hash
            }

        # 8. 返回完整结果
        return {
            "found": True,
            "file": rel_file_path,
            "function_name": function_name,
            "line_number": target_line_number,
            "commit_hash": commit_hash,
            "commit_time": target_commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "commit_msg": target_commit.message.strip(),
            "diff": diff_content
        }

    except Exception as e:
        return {"error": str(e)}


# --- 测试代码 ---
if __name__ == "__main__":
    # 配置参数
    REPO_DIR = "/path/to/your/linux_kernel_or_project"  # 替换为实际 Git 仓库路径
    FILE_PATH = "src/core/main.c"  # 替换为实际 C 文件路径
    PROBLEM_TIME = "2025-11-20 11:19:30"  # 问题发现时间
    PROBLEM_LINE = 158  # 报错的具体行号

    # 运行分析
    result = analyze_code_context(REPO_DIR, FILE_PATH, PROBLEM_TIME, PROBLEM_LINE)

    if result.get("error"):
        print(f"❌ 错误: {result['error']}")
    elif not result.get("found"):
        print(f"⚠️ 跳过: {result['reason']}")
        print(f"   File: {result['file']} @ Commit: {result['commit_hash'][:8]}")
    else:
        print("=" * 50)
        print(f"✅ 定位成功")
        print(f"所在函数: {result['function_name']}")
        print(f"文件路径: {result['file']} (Line {result['line_number']})")
        print(f"Commit  : {result['commit_hash']} ({result['commit_time']})")
        print(f"Message : {result['commit_msg']}")
        print("-" * 50)
        print("Diff 片段 (前500字符):")
        print(result['diff'][:500] + "\n..." if len(result['diff']) > 500 else result['diff'])
        print("=" * 50)