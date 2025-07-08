import os
import tree_sitter_c
from tree_sitter import Language, Parser

"""
pip show tree_sitter
Name: tree-sitter
Version: 0.23.2

pip install tree-sitter-c==0.23.2
"""

class CFunctionExtractor:
    def __init__(self):
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser(self.language)

    def extract_functions(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        lines = code.splitlines()
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        functions = []

        def traverse(node):
            if node.type == 'function_definition':
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                func_lines = lines[start_line:end_line + 1]

                # 去除空行，保留注释和代码
                non_empty_lines = [l for l in func_lines if l.strip() != '']

                # 提取函数名
                func_name = None
                for child in node.children:
                    if child.type == 'declarator':
                        for sub in child.children:
                            if sub.type == 'identifier':
                                func_name = code[sub.start_byte:sub.end_byte]
                                break

                functions.append({
                    'name': func_name,
                    'start_line': start_line + 1,
                    'end_line': end_line + 1,
                    'code': "\n".join(non_empty_lines),
                    'ast_node': node
                })

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return functions

from pprint import pprint  # 用于美观打印 dict

# === 新增：AST 转 Python dict 的辅助函数 ===
def node_to_dict(node, code_bytes):
    d = {
        'type': node.type,
        'start_point': node.start_point,
        'end_point': node.end_point,
    }
    if len(node.children) == 0:
        d['text'] = code_bytes[node.start_byte:node.end_byte].decode('utf-8')
    else:
        d['children'] = [node_to_dict(child, code_bytes) for child in node.children]
    return d

# === 以下保留原结构，只做最小修改 ===
if __name__ == '__main__':

    test_c_code = '''
    #include <stdio.h>

    // 全局变量
    int global_var = 100;

    // 结构体定义
    struct Point {
        int x;
        int y;
    };

    // 普通函数
    int add(int a, int b) {
        return a + b;
    }

    // 带注释的函数
    int subtract(int a, int b) {
        // 计算差值
        return a - b;
    }
    '''

    test_file_path = 'temp_test.c'
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_c_code)

    extractor = CFunctionExtractor()
    funcs = extractor.extract_functions(test_file_path)

    with open(test_file_path, 'rb') as f:
        code_bytes = f.read()

    for func in funcs:
        print(f"Function: {func['name']}")
        print(f"Lines: {func['start_line']} - {func['end_line']}")
        print("Code:\n" + func['code'])

        # ✅ 新增：打印该函数的 AST dict
        ast_dict = node_to_dict(func['ast_node'], code_bytes)
        pprint(ast_dict)

        print("=" * 40)