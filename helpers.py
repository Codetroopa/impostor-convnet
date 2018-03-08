import os

# When we save a file, we don't an empty file_path to result in saving to root
def prefix_to_file(file_path):
    path = os.path.dirname(file_path)
    if path is "":
        path = "."
    return path
