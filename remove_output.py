"""
Usage: python remove_output.py notebook.ipynb [ > without_output.ipynb ]
Modified from remove_output by Minrk

"""
import os
import sys
from nbformat import read, write, NO_CONVERT


def remove_outputs(nb):
    """remove the outputs from a notebook"""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []

if __name__ == '__main__':
    fname = sys.argv[1]
    with open(fname, 'r') as f:
        nb = read(f, NO_CONVERT)
    remove_outputs(nb)
    base, ext = os.path.splitext(fname)
    new_ipynb = f"{base}_removed{ext}"
    with open(new_ipynb, 'w', encoding='utf8') as f:
        write(nb, f)
    print(f"wrote {new_ipynb}")