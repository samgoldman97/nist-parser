""" count_adducts.py"""
from collections import Counter
import pandas as pd


labels = "processed_data/labels.tsv"
df = pd.read_csv(labels, sep="\t")
counts = Counter(df['ionization'].values)

out = sorted([(i, v)
              for i, v in counts.items()],
             key=lambda x: x[1])[::-1]

out = [(i, j) for i, j in out if i[-1] == "+"]


out = [f"{i} {j}" for i, j in out if j > 100]
print("\n".join(out))

valid = ["[M+H]+",
         "[M+Na]+",
         "[M+K]+",
         "[M+H-H2O]+",
         "[M+NH4]+",
         "[M+H-2H2O]+"]
