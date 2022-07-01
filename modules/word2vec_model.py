import gensim
import numpy as np
import traceback
from tqdm import tqdm

GOOGLE_PATH = "/home/gianluca/bootcamp/lighthouse_data_notes/week_8/day_3/google_news_vector/GoogleNews-vectors-negative300.bin"

def create_model(list_tokens):
    """Creates a word to vec model given an iterable of lists of tokens. Intersects the model
    on Googles Word2Vec model, and trains on given list of tokens.
    Returns: <gensim.models.Word2Vec> | None
    """
    print("Instantiating new word2vec model ..")
    model = gensim.models.Word2Vec(sentences=list_tokens, vector_size=300)
    print("Model created from new tokens.")
    print("Intersecting model with Google Word2Vec...")
    model.wv.vectors_lockf = np.ones(len(model.wv))
    try:
        model.wv.intersect_word2vec_format(GOOGLE_PATH, lockf=1.0, binary=True)
    except:
        print("Error while intersecting")
        return 
    
    print("Training the model")
    model.train(tqdm(list_tokens), total_examples=model.corpus_count, epochs=10)
    return model


def save_model(model, path):
    """Saves the model to a file path given by user
    Returns: bool
    """
    try:
        print("Saving model...")
        with open(path, "wb") as f:
            model.save(f)
        print(f"Model saved to {path}")
        return True
    except:
        traceback.print_exc()
        print(f"Error while trying to save the model at {path}")
        return False
    
def load_model(path):
    """Loads the Word2Vec model from the given path
    Returns: <gensim.models.Word2Vec> | None
    """
    try:
        return gensim.models.Word2Vec.load(path)
    except:
        print(f"Error while trying to load model at {path}")
        return