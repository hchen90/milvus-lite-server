#!/usr/bin/env python3
"""
测试嵌入API端点的脚本
"""

import requests
import json
import sys
import jwt
from datetime import datetime, timedelta, timezone


# JWT 配置（应与服务器配置一致）
JWT_SECRET_KEY = "your-super-secret-jwt-key-change-this-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 30


def create_test_token(username: str = "test_user", expires_minutes: int = 30) -> str:
    """创建测试用的 JWT 令牌"""
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    
    payload = {
        "sub": username,
        "email": f"{username}@example.com", 
        "full_name": f"Test User {username.title()}",
        "exp": expire
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def get_auth_headers():
    """获取包含JWT令牌的认证请求头"""
    token = create_test_token("embedding_test_user")
    return {"Authorization": f"Bearer {token}"}


def test_single_embedding():
    """测试单个嵌入向量生成API"""
    print("=== 测试单个嵌入向量生成 ===")
    
    url = "http://localhost:8000/api/v1/embedding"
    data = {
        "content": "这是一个测试文本，用于生成嵌入向量。"
    }
    
    # 添加JWT认证头
    headers = get_auth_headers()
    headers["Content-Type"] = "application/json"
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功！")
            print(f"内容: {result['content']}")
            print(f"向量维度: {result['dimension']}")
            print(f"向量前5个值: {result['embedding'][:5]}")
        else:
            print(f"❌ 失败！状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")


def test_chunked_embeddings():
    """测试分块嵌入向量生成API"""
    print("\n=== 测试分块嵌入向量生成 ===")
    
    url = "http://localhost:8000/api/v1/embeddings"
    # 使用较长的文本来触发分块
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程的模拟。
    人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
    """ * 10  # 重复10次以确保触发分块
    
    data = {
        "content": long_text.strip()
    }
    
    # 添加JWT认证头
    headers = get_auth_headers()
    headers["Content-Type"] = "application/json"
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 成功！")
            print(f"内容长度: {len(result['content'])}")
            print(f"总分块数: {result['total_chunks']}")
            print(f"第一块文本: {result['embeddings'][0]['text'][:50]}...")
            print(f"第一块向量维度: {len(result['embeddings'][0]['embedding'])}")
        else:
            print(f"❌ 失败！状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")


def test_health_check():
    """测试健康检查端点"""
    print("\n=== 测试健康检查 ===")
    
    url = "http://localhost:8000/health"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 服务运行正常！")
            print(f"状态: {result['status']}")
            print(f"应用名称: {result['app_name']}")
            print(f"版本: {result['version']}")
        else:
            print(f"❌ 健康检查失败！状态码: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 无法连接到服务器: {e}")
        print("请确保服务器正在运行（python -m app.main 或 docker-compose up）")
        return False
    
    return True


def main():
    """主函数"""
    print("开始测试嵌入API端点...")
    print("🔐 创建JWT认证令牌...")
    
    # 创建并显示JWT令牌信息
    token = create_test_token("embedding_test_user")
    print(f"   令牌已创建: {token[:30]}...")
    
    # 先检查服务器是否运行
    if not test_health_check():
        sys.exit(1)
    
    # 测试嵌入API
    test_single_embedding()
    test_chunked_embeddings()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
