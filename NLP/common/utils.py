import numpy as np
from bidict import bidict


def preprocess(text: str):
    """
    Args:
        text: str text for creating corpus
    Returns:
        corpus: np.ndarray corpus created from text
        word_to_dict: bidict[key, int] dict[word, id]
    """
    text = text.lower()  # 最初の文字も他の文字と同様に扱うために小文字に変換
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id: bidict = bidict({})
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id

    corpus = np.array([id for id in word_to_id.values()])

    return corpus, word_to_id
