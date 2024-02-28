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


def create_co_matrix(
    corpus: np.ndarray, vocab_size: int, window_size: int = 1
) -> np.ndarray:
    """
    create co-occurrence matrix
    Args:
        corpus:
        vocab_size:
        window_size:

    Returns:
        co-matrix: np.ndarray
    """
    corpus_size = len(corpus)
    co_matrix = np.empty((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        # 自分を数えないようにするために、indexは1から
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[right_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix
