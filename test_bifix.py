import sys
sys.path.insert(0, ".")
from app.subtitle import merge_broken_contractions, merge_number_runs

w2 = [
    {"start": 0.0, "end": 0.2, "word": "is", "conf": 0.9},
    {"start": 0.2, "end": 0.4, "word": "n't", "conf": 0.9},
    {"start": 0.5, "end": 0.7, "word": "here", "conf": 0.9},
]
out, n = merge_broken_contractions(w2)
print(out, "merged:", n)

nums = [
    {"start": 1.0, "end": 1.2, "word": "tres", "conf": 0.9},
    {"start": 1.25, "end": 1.5, "word": "mil", "conf": 0.9},
    {"start": 1.6, "end": 2.0, "word": "millones", "conf": 0.9},
    {"start": 2.4, "end": 2.6, "word": "años", "conf": 0.9},
]
out2, n2 = merge_number_runs(nums, "es")
print(out2, "merged:", n2)
