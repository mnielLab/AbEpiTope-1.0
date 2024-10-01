import pickle

def load_pickle_file(infile_path):
    with open(infile_path, "rb") as infile:
        pickle_data = pickle.load(infile)
    return pickle_data