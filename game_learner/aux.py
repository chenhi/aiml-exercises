def log(l: str, print_out=True) -> str:
    if print_out:
        print(l)
    return str(l) + "\n"

# span n means length 2n+1
def smoothing(vals: list, span: int):
    out = []
    for j in range(len(vals)):
        total = 0.
        num = 0
        for k in range(-span, span+1):
            if j + k >= 0 and j + k < len(vals):
                total += vals[j+k]
                num += 1
        out.append(total/num)
    return out