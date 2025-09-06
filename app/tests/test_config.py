#!/usr/bin/env python3
"""
配置测试脚本
用于验证config.py是否正常工作
"""

import unittest
from core.config import config


class TestConfig(unittest.TestCase):
    """配置测试类"""
    
    def test_config_loading(self):
        """测试配置加载"""
        self.assertIsNotNone(config.APP_NAME)
        self.assertIsNotNone(config.VERSION)
        self.assertIsInstance(config.DEBUG, bool)
        
    def test_server_address(self):
        """测试服务器地址配置"""
        server_address = config.get_server_address()
        self.assertIsNotNone(server_address)
        self.assertIn(":", server_address)
        
    def test_config_validation(self):
        """测试配置验证"""
        is_valid = config.validate_config()
        self.assertTrue(is_valid)
        
    def test_log_level(self):
        """测试日志级别"""
        self.assertIn(config.LOG_LEVEL, ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])




