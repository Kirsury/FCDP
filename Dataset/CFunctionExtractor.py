from tree_sitter import Language, Parser

class CFunctionExtractor:
    def __init__(self, language_so_path='build/my-languages.so'):
        self.language = Language(language_so_path, 'c')
        self.parser = Parser()
        self.parser.set_language(self.language)

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
                    'code': "\n".join(non_empty_lines)
                })

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return functions
