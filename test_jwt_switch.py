#!/usr/bin/env python3
"""
JWT开关功能测试脚本

测试在JWT启用和禁用状态下，API接口的访问行为
"""
import os
import sys
import requests
import json
from datetime import datetime

# 测试服务器地址
BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None, headers=None, description=""):
    """测试API端点"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n测试: {description or endpoint}")
    print(f"URL: {method} {url}")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            print(f"不支持的HTTP方法: {method}")
            return False
        
        print(f"状态码: {response.status_code}")
        try:
            response_data = response.json()
            print(f"响应: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
        except:
            print(f"响应 (非JSON): {response.text}")
        
        return response.status_code < 400
    
    except requests.exceptions.ConnectionError:
        print(f"❌ 连接失败: 请确保服务器在 {BASE_URL} 上运行")
        return False
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def main():
    """主测试函数"""
    print(f"🧪 JWT开关功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标服务器: {BASE_URL}")
    print("=" * 50)
    
    # 测试基本端点
    print("\n🔍 测试基本端点 (不需要JWT)")
    test_endpoint("/", description="根路径")
    test_endpoint("/health", description="健康检查")
    
    # 测试不带JWT的API访问
    print("\n🔓 测试不带JWT的API访问")
    test_endpoint("/api/v1/auth/profile", description="获取用户信息 (无JWT)")
    test_endpoint("/api/v1/auth/verify", method="POST", description="验证令牌 (无JWT)")
    
    # 测试向量搜索API (表单数据)
    print("\n🔍 测试向量搜索API (无JWT)")
    test_endpoint("/api/v1/documents/search?query=test&limit=5", description="搜索文档 (GET方式)")
    
    # 测试向量搜索API (JSON数据)
    search_data = {
        "query": "测试查询文本",
        "limit": 3
    }
    test_endpoint("/api/v1/documents/search", method="POST", data=search_data, description="搜索文档 (POST方式)")
    
    # 测试嵌入API
    print("\n🔧 测试嵌入API (无JWT)")
    embedding_data = {
        "content": "这是一个测试文本，用于生成嵌入向量"
    }
    test_endpoint("/api/v1/embedding", method="POST", data=embedding_data, description="生成嵌入向量")
    
    # 测试文档保存API (通常需要认证，但现在应该允许匿名访问)
    print("\n💾 测试文档保存API (无JWT)")
    save_data = {
        "post_id": "test-001",
        "title": "测试文档",
        "content": "这是一个测试文档的内容，用于测试JWT开关功能。"
    }
    test_endpoint("/api/v1/documents/json", method="POST", data=save_data, description="保存文档 (JSON方式)")
    
    print("\n" + "=" * 50)
    print("✅ 测试完成!")
    print("\n💡 提示:")
    print("- 如果JWT_ENABLED=False，所有API应该允许匿名访问")
    print("- 如果JWT_ENABLED=True，需要JWT的API应该返回401错误")
    print("- 可以通过设置环境变量 JWT_ENABLED=false 来禁用JWT验证")

if __name__ == "__main__":
    main()