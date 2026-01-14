import json

notebook_path = '/home/code/research/kernels/smartmanoj_santa-claude/santa-claude.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.startswith('%%writefile a.cpp'):
            print("Found a.cpp")
            content = source.replace('%%writefile a.cpp\n', '')
            with open('tree_packer.cpp', 'w') as f_out:
                f_out.write(content)
        elif source.startswith('%%writefile bp.cpp'):
            print("Found bp.cpp")
            content = source.replace('%%writefile bp.cpp\n', '')
            with open('bp.cpp', 'w') as f_out:
                f_out.write(content)
