print(">>> starting debug script")

import evaluation.metrics as m

print("Loaded from:", m.__file__)
print("Dir:", dir(m))
with open(m.__file__, encoding="utf-8") as f:
    print("Source:\n", f.read())
