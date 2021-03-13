import pickle


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, fname):
    with open(fname, "wb") as f:
        return pickle.dump(obj, f)
