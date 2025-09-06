import logging
from .embedding import get_embeddings_from_content, get_embedding_from_content
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema, MilvusException, model

def setup_milvus_collection(client: MilvusClient, collection_name: str, collection_dimension: int, force: bool) -> bool:
    """
    Set up the Milvus collection for blog posts.
    
    Args:
        client (MilvusClient): Initialized Milvus client
        collection_name (str): Name of the collection to create
        collection_dimension (int): Dimension of the collection
        force (bool): If True, force recreate the collection if it exists
    """
    if client.has_collection(collection_name) and not force:
        logging.info(f"Collection {collection_name} already exists.")
        return False

    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="post_id", dtype=DataType.VARCHAR, max_length=100, description="Post ID"),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200, description="Post Title"),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535, description="Post Content"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=collection_dimension, description="Text Embedding"),
    ]
    schema = CollectionSchema(fields=fields, description="Blog Posts Embeddings")
    
    # Check if collection exists and drop if needed
    if client.has_collection(collection_name):
        logging.info(f"Collection {collection_name} already exists. Dropping it...")
        client.drop_collection(collection_name)
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        dimension=collection_dimension,
        schema=schema,
    )
    logging.info(f"Collection {collection_name} created successfully.")

    return True

def create_index(client: MilvusClient, collection_name: str) -> None:
    """
    Create an index on the collection for efficient searching.
    
    Args:
        client (MilvusClient): Initialized Milvus client
        collection_name (str): Name of the collection to create the index on
    """
    INDEX_TYPE = "FLAT"
    METRIC_TYPE = "L2"
    INDEX_NAME = "embedding_index"
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type=METRIC_TYPE,
        index_type=INDEX_TYPE,
        index_name=INDEX_NAME,
        params={"nlist": 128}
    )

    try:
        index_info = client.describe_index(
            collection_name=collection_name,
            index_name=INDEX_NAME
        )
        if index_info:
            logging.info(f"Index '{INDEX_NAME}' exists on collection '{collection_name}': {index_info}")
            # If index already exists, we can skip creating it
            return
    except MilvusException as e:
        if "index doesn't exist" in str(e).lower(): # Check if the error is about index not existing
            logging.info(f"Index '{INDEX_NAME}' does not exist on collection '{collection_name}'.")
        else:
            logging.error(f"An error occurred: {e}")
            return
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return

    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    logging.info(f"Index created successfully.")

def insert_data(client: MilvusClient, collection_name: str, post_id: str, post_title: str, post: str) -> bool:
    """
    Insert data into the Milvus collection.
    
    Args:
        client (MilvusClient): Initialized Milvus client
        collection_name (str): Name of the collection to insert data into
        data (list): List of data entries to insert
    """
    if not post:
        logging.warning("No content provided for insertion.")
        return False
    
    # Embedding content
    embeddings = get_embeddings_from_content(post)
    if not embeddings:
        logging.error(f"No embedding generated for post ID {post_id}. Skipping.")
        return False
    logging.info(f"Post ID {post_id} embeddings generated successfully.")

    # Prepare data for insertion
    data_to_insert = []

    for embedding in embeddings:
        data_to_insert.append({
            "post_id": post_id,
            "title": post_title,
            "content": embedding.get("text", ""),
            "embedding": embedding.get("embedding", [])
        })
        logging.debug(f"Embedding for post ID {post_id} generated successfully.")
    if len(data_to_insert) > 0:
        res = client.insert(
            collection_name=collection_name,
            data=data_to_insert
        )
        logging.info(f"Inserted {len(data_to_insert)} embeddings for post ID {post_id}, result: {res}")
    logging.info(f"Post ID {post_id} inserted into Milvus collection {collection_name}.")
    return True

def search_data(client: MilvusClient, collection_name: str, text: str, limit: int = 5) -> list:
    """
    Search for similar entries in the Milvus collection based on the provided text.
    
    Args:
        client (MilvusClient): Initialized Milvus client
        collection_name (str): Name of the collection to search
        text (str): The content to search for
        limit (int): Number of results to return (default: 5)
    
    Returns:
        list: List of search results with content and similarity scores
    """
    if not text or not isinstance(text, str):
        logging.error("Invalid query content provided.")
        return []

    try:
        # Check if collection exists first (before generating embedding)
        if not client.has_collection(collection_name):
            logging.error(f"Collection {collection_name} does not exist.")
            return []

        # Use the project's embedding function to encode the query
        query_embedding = get_embedding_from_content(text)
        if query_embedding is None or len(query_embedding) == 0:
            logging.error("Failed to generate embedding for the query text.")
            return []
        
        # Prepare query vectors for search
        query_vectors = [query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding]
        
        logging.info(f"Generated query embedding for text: {text[:100]}...")

        # Perform search
        res = client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=["post_id", "title", "content"]
        )

        # Process search results
        search_results = []
        if res and len(res) > 0:
            for hits in res:
                for hit in hits:
                    search_results.append({
                        "id": hit.get("id"),
                        "post_id": hit.get("post_id"),
                        "title": hit.get("title"),
                        "content": hit.get("content"),
                        "distance": hit.get("distance"),
                        "score": 1 / (1 + hit.get("distance", 0))  # Convert distance to similarity score
                    })
            
            logging.info(f"Found {len(search_results)} similar entries for query: {text[:50]}...")
        else:
            logging.info(f"No results found for query: {text[:50]}...")

        return search_results

    except MilvusException as e:
        logging.error(f"Milvus error during search: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during search: {e}")
        return []