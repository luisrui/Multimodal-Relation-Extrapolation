def read_clean_line(path):
    with open(path, 'r') as f:
        data = []
        for line in f.readlines():
            if line[-1] == '\n':
                data.append(line[:-1])
            else:
                data.append(line)
    return data