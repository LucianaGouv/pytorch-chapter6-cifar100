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
backup = nb_path + '.bak.sanitize.' + time.strftime('%Y%m%d%H%M%S')
shutil.copy(nb_path, backup)
print('Backup written to', backup)

nb = nbformat.read(nb_path, as_version=4)
# Remove 'id' fields from cells to avoid nbconvert validation issues
changed = False
for cell in nb.cells:
    if isinstance(cell, dict) and 'id' in cell:
        del cell['id']
        changed = True

if changed:
    nbformat.write(nb, nb_path)
    print('Removed stray cell id fields and updated notebook')
else:
    print('No cell id fields found; no change needed')

# Execute notebook in-place
python_exec = sys.executable
cmd = [python_exec, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', nb_path,
       '--ExecutePreprocessor.timeout=2400', '--ExecutePreprocessor.kernel_name=pytorch-ch6-venv']
print('Executing notebook (this may take several minutes):')
print(' '.join(cmd))
subprocess.check_call(cmd)
print('Notebook executed and saved; high-DPI figures should be regenerated in the figures/ directory.')
