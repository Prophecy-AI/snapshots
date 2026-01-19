import json
import os

# Read the why-not notebook
with open('/home/code/research/kernels/jazivxt_why-not/why-not.ipynb', 'r') as f:
    nb = json.load(f)

# Find the cell with bbox3.cpp
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if '%%writefile bbox3.cpp' in source:
            # Extract the C++ code (skip the first line with %%writefile)
            lines = source.split('\n')
            cpp_code = '\n'.join(lines[1:])
            
            with open('/home/code/experiments/009_bbox3_optimizer/bbox3.cpp', 'w') as f:
                f.write(cpp_code)
            print(f"Extracted bbox3.cpp ({len(cpp_code)} bytes)")
            break
else:
    print("Could not find bbox3.cpp in notebook")
