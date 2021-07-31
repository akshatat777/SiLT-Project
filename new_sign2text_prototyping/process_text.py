
path_to_wikipedia = "wikipedia2text-extracted.txt"  # update this path if necessary
with open(path_to_wikipedia, "rb") as f:
    wikipedia = f.read().decode().lower()

print(len(wikipedia))