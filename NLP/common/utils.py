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

    # valuesメソッドを使うと正しくない [id for id in word_to_id.values()]
    corpus = np.array([word_to_id[w] for w in words])

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
    # emptyでは不定の値が入る可能性があるのでzerosにする
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        # 自分を数えないようにするために、indexは1から
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-4):
    nx = x / (np.sqrt(np.sum(np.square(x))) + eps)
    ny = y / (np.sqrt(np.sum(np.square(y))) + eps)
    return nx @ ny


def most_similarity(
    query: str, word_to_id: bidict, word_matrix: np.ndarray, top: int = 5
):
    if query not in word_to_id:
        ValueError(f"{query} is not found")

    print("\n[query] ", query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # calculate cos similarity
    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 + similarity).argsort():
        if word_to_id.inverse[i] == query:
            continue
        print(f"{word_to_id.inverse[i]}: {similarity[i]}")
        count += 1
        if count >= top:
            return
