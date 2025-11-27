import nbformat
import sys
import json

def fix_and_validate_notebook(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        with open(file_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
            nbformat.validate(nb)
        print(f"Notebook '{file_path}' is now valid.")
    except Exception as e:
        print(f"Error fixing and validating notebook '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_and_validate_notebook.py <notebook_path>", file=sys.stderr)
        sys.exit(1)
    fix_and_validate_notebook(sys.argv[1])
