
def read_file(fp, skip_header=False, split=False):
    with open(fp, "r") as f:
        lines = f.readlines()
        if skip_header:
            lines = lines[1:]
    if split:
        items = [l.strip().split("\t") for l in lines]
    else:
        items = [l.strip() for l in lines]
    return items