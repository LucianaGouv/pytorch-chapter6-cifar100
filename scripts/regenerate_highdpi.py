#!/usr/bin/env python3
import nbformat
import shutil
import time
import subprocess
import sys
import os

nb_path = os.path.join('notebooks', 'Cifar100.ipynb')
if not os.path.exists(nb_path):
    print('Notebook not found:', nb_path)
    sys.exit(1)

# Backup
backup = nb_path + '.bak.' + time.strftime('%Y%m%d%H%M%S')
shutil.copy(nb_path, backup)
print('Backup written to', backup)

nb = nbformat.read(nb_path, as_version=4)
cell_code = '''# >>> AUTO-INSERTED: For publication-quality figures (high DPI)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.figsize'] = (6,4)
print('High-DPI defaults set: figure.dpi=', mpl.rcParams['figure.dpi'], 'savefig.dpi=', mpl.rcParams['savefig.dpi'])
'''
cell = nbformat.v4.new_code_cell(cell_code)
# Insert at top
nb.cells.insert(0, cell)
nbformat.write(nb, nb_path)
print('Prepended high-DPI cell to', nb_path)

# Execute notebook in-place
python_exec = sys.executable
cmd = [python_exec, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', nb_path,
       '--ExecutePreprocessor.timeout=2400', '--ExecutePreprocessor.kernel_name=pytorch-ch6-venv']
print('Executing notebook (this may take several minutes):')
print(' '.join(cmd))
subprocess.check_call(cmd)
print('Notebook executed and saved; figures should be regenerated with higher DPI.')
