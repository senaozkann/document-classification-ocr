"""Açık bir Colab oturumunda (Drive zaten bağlıyken) GitHub'daki bir notebook'un kod hücrelerini sırayla çalıştırır.

Kullanım (Colab hücresi):
    import urllib.request; exec(urllib.request.urlopen(RAW_URL_OF_THIS_FILE).read().decode())
    calistir("notebooks/10_son_model.ipynb", ref="main")

`!` ile başlayan kurulum satırları kabukta, `%%writefile` hücreleri dosyaya yazılarak, diğer hücreler aynı
global ad alanında çalışır. Böylece yeni bir oturum açıp Drive iznini tekrar vermek gerekmez.
"""
import json
import subprocess
import urllib.request

REPO_RAW = "https://raw.githubusercontent.com/senaozkann/document-classification-ocr"


def calistir(nb_path, ref="main", skip_mount=True, g=None):
    g = g if g is not None else globals()
    nb = json.load(urllib.request.urlopen(f"{REPO_RAW}/{ref}/{nb_path}"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if src.startswith("%%writefile"):
            first, body = src.split("\n", 1)
            path = first.split(maxsplit=1)[1]
            with open(path, "w") as f:
                f.write(body)
            print(f"[{i}] yazıldı: {path}")
            continue
        code_lines = []
        for line in src.splitlines():
            if line.startswith("!"):
                subprocess.run(line[1:], shell=True, check=False)
            else:
                code_lines.append(line)
        code = "\n".join(code_lines)
        if skip_mount:
            code = code.replace('from google.colab import drive\ndrive.mount("/content/drive")', "")
        if code.strip():
            print(f"[{i}] çalışıyor…")
            exec(compile(code, f"{nb_path}#hücre{i}", "exec"), g)
    print("BITTI:", nb_path)
