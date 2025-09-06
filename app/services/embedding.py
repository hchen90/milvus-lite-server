
import logging
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def get_model() -> SentenceTransformer:
    """Load and return the SentenceTransformer model."""
    try:
        model = SentenceTransformer(MODEL_NAME)
        logging.debug(f"Model {MODEL_NAME} loaded successfully.")
        return model
    except Exception as e:
        logging.debug(f"Error loading model {MODEL_NAME}: {e}")
        raise e

def get_tokenizer_info(article: str) -> tuple:
    """
    Get the number of tokens in the article using the SentenceTransformer tokenizer.
    Args:
        article (str): The article text to tokenize.
    Returns:
        tuple: Number of tokens in the article and the maximum tokens allowed by the tokenizer.
    """
    model = get_model()
    encoded_input = model.tokenizer(article, return_tensors='pt', truncation=False)
    num_tokens = len(encoded_input['input_ids'][0])
    max_tokens = model.tokenizer.model_max_length
    logging.debug(f"Number of tokens: {num_tokens}, Max tokens: {max_tokens}")
    return num_tokens, max_tokens

def get_embedding_from_content(content: str) -> list:
    """
    Get the embedding vector for the given content using the SentenceTransformer model.
    Args:
        content (str): The content to embed.
    Returns:
        list: The embedding vector for the content.
    """
    if not content:
        return []
    
    model_instance = get_model()
    embeddings = model_instance.encode(content, show_progress_bar=True)
    return embeddings

def get_embeddings_from_content(post: str) -> list:
    """
    Get the embedding vectors for a single post.
    Args:
        post (str): The post to embed.
    Returns:
        list: A list of embedding vectors for the post.
    """
    if not post:
        return []
    length = len(post)
    if length == 0:
        return []

    model_instance = get_model()
    num_tokens, max_tokens = get_tokenizer_info(post)
    count = (num_tokens - 1) // max_tokens + 1 if num_tokens > max_tokens else 1
    if count <= 0:
        return []
    chunk_size = length // count

    logging.info(f"Post length: {length}, Number of tokens: {num_tokens}, Max tokens: {max_tokens}, Chunks: {count}, Chunk size: {chunk_size}")
    
    embeddings = []

    for i in range(count):
        text = post[i * chunk_size: (i + 1) * chunk_size]
        if len(text) == 0:
            continue
        embedding = model_instance.encode(text, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
        if embedding is None or len(embedding) == 0:
            continue
        logging.debug(f"Embedding for chunk {i + 1}/{count} generated successfully.")
        embeddings.append({
            "text": text,
            "embedding": embedding.tolist()
        })

    logging.debug(f"Total embeddings generated: {len(embeddings)}")
    return embeddings