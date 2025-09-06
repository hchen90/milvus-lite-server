import unittest
import services.embedding as embedding_service

class TestEmbeddingService(unittest.TestCase):
    """Embedding服务测试类"""
    
    def test_get_tokenizer_info(self):
        """测试获取分词器信息"""
        article = "This is a test article."
        num_tokens, max_tokens = embedding_service.get_tokenizer_info(article)
        self.assertIsInstance(num_tokens, int)
        self.assertIsInstance(max_tokens, int)
        self.assertGreater(num_tokens, 0)
        self.assertGreater(max_tokens, 0)
    
    def test_get_embedding_from_content(self):
        """测试从内容获取嵌入向量"""
        content = "This is a test content."
        embedding = embedding_service.get_embedding_from_content(content)
        self.assertGreater(len(embedding), 0)
    
    def test_get_embeddings_from_content(self):
        """测试从单个帖子获取嵌入向量列表"""
        post = "This is a test post. " * 50  # 长文本以测试分块
        embeddings = embedding_service.get_embeddings_from_content(post)
        self.assertGreater(len(embeddings), 0)
        for item in embeddings:
            self.assertIn("text", item)
            self.assertIn("embedding", item)
            self.assertGreater(len(item["embedding"]), 0)
