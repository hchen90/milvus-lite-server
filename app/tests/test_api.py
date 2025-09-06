#!/usr/bin/env python3
"""
应用逻辑测试脚本
用于测试应用的基本逻辑功能（不依赖HTTP客户端）
"""

import unittest
from unittest.mock import patch, MagicMock

class TestApplicationLogic(unittest.TestCase):
    """应用逻辑测试类"""
    
    def setUp(self):
        """测试前的设置"""
        # 模拟配置以避免真实配置文件的依赖
        self.mock_config = MagicMock()
        self.mock_config.APP_NAME = "Milvus Lite Server"
        self.mock_config.VERSION = "0.1.0"
        self.mock_config.DEBUG = False
        self.mock_config.LOG_LEVEL = "INFO"
        self.mock_config.HOST = "0.0.0.0"
        self.mock_config.PORT = 8000
        self.mock_config.get_server_address.return_value = "localhost:8000"
        self.mock_config.validate_config.return_value = True
    
    def test_app_creation(self):
        """测试FastAPI应用创建"""
        from main import app
        # 只验证应用对象存在和基本属性
        self.assertIsNotNone(app)
        self.assertIsNotNone(app.title)
        self.assertIsNotNone(app.version)
        self.assertIsInstance(app.debug, bool)
    
    def test_main_function_with_valid_config(self):
        """测试main函数在有效配置下的行为"""
        with patch('main.config', self.mock_config):
            with patch('main.uvicorn.run') as mock_uvicorn:
                from main import main
                
                # 运行main函数，但不实际启动服务器
                main()
                
                # 验证uvicorn.run被调用
                mock_uvicorn.assert_called_once_with(
                    "main:app",
                    host="0.0.0.0",
                    port=8000,
                    reload=False,
                    log_level="info"
                )
    
    def test_main_function_with_invalid_config(self):
        """测试main函数在无效配置下的行为"""
        # 设置配置验证失败
        self.mock_config.validate_config.return_value = False
        
        with patch('main.config', self.mock_config):
            with patch('main.uvicorn.run') as mock_uvicorn:
                from main import main
                
                # 运行main函数
                main()
                
                # 验证uvicorn.run没有被调用
                mock_uvicorn.assert_not_called()
    
    def test_route_functions_exist(self):
        """测试路由函数是否存在"""
        with patch('main.config', self.mock_config):
            import main
            
            # 检查路由函数是否存在
            self.assertTrue(hasattr(main, 'root'))
            self.assertTrue(hasattr(main, 'health_check'))
            self.assertTrue(hasattr(main, 'get_config'))
            self.assertTrue(callable(main.root))
            self.assertTrue(callable(main.health_check))
            self.assertTrue(callable(main.get_config))



