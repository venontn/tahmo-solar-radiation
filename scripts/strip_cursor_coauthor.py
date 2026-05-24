import sys
data = sys.stdin.read()
out = "".join(line for line in data.splitlines(keepends=True) if "Co-authored-by: Cursor" not in line)
sys.stdout.write(out)
