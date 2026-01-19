import json
import os

notebook_path = '/home/code/research/kernels/smartmanoj_santa-claude/santa-claude.ipynb'
output_dir = '/home/code/experiments/001_baseline'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.startswith('%%writefile'):
            filename = source.split()[1]
            content = source.split('\n', 1)[1]
            
            # Rename a.cpp to tree_packer_v21.cpp
            if filename == 'a.cpp':
                filename = 'tree_packer_v21.cpp'
            
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w') as f_out:
                f_out.write(content)
            print(f"Extracted {filename}")
