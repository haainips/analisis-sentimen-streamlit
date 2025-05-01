import re

def load_slang_dictionary(filepath):
    clear_slangs = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as slangs:
        for newlines in slangs:
            strip_re = newlines.strip("\n")
            split = re.split(r'[:]', strip_re)
            clear_slangs.append(split)

    slangs = [[k.strip(), v.strip()] for k, v in clear_slangs]
    return {key: values for key, values in slangs}
