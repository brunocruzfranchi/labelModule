from os import walk


def get_all_json_files(path):
    filenames = next(walk(path), (None, None, []))[2]  # [] if no file
    return filenames
