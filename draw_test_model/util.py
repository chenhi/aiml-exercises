from io import StringIO

def log(l: str) -> str:
    print(l)
    return str(l) + "\n"

def tf_summary_string(m) -> str:
    string_io = StringIO()
    m.summary(print_fn=lambda x: string_io.write(f"{x}\n"))
    return string_io.getvalue()