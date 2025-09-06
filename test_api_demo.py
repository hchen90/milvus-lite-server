#!/usr/bin/env python3
"""
测试向量数据库API接口
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_save_document():
    """测试保存文档接口（表单形式）"""
    print("📝 测试保存文档接口（表单形式）...")
    
    # 测试数据
    test_data = {
        "post_id": "doc001",
        "title": "人工智能技术的发展",
        "content": """
人工智能（AI）是计算机科学的一个分支，它试图让机器模拟人类智能的行为。
随着深度学习和神经网络技术的发展，AI在许多领域都取得了重大突破。

机器学习是AI的核心技术之一，它允许计算机从数据中学习和改进，而无需明确编程。
深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的数据模式。

在自然语言处理方面，大型语言模型如GPT系列展现了惊人的能力。
这些模型能够理解和生成人类语言，并在各种任务中表现出色。

计算机视觉是另一个重要的AI应用领域，它让机器能够"看"和理解图像。
从图像识别到自动驾驶，计算机视觉技术正在改变我们的生活方式。
        """.strip()
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/documents", data=test_data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()
    
    return response.status_code == 200

def test_save_document_json():
    """测试保存文档接口（JSON形式）"""
    print("📝 测试保存文档接口（JSON形式）...")
    
    # 测试数据
    test_data = {
        "post_id": "doc002", 
        "title": "云计算技术概述",
        "content": """
云计算是一种通过互联网提供计算资源的模式，包括服务器、存储、数据库、网络、软件等。
它使得用户可以按需访问这些资源，而无需拥有和维护物理基础设施。

云计算的主要服务模式包括：
1. 基础设施即服务（IaaS）：提供虚拟化的计算资源
2. 平台即服务（PaaS）：提供开发和部署应用程序的平台
3. 软件即服务（SaaS）：提供基于云的软件应用程序

云计算的优势包括成本效益、可扩展性、灵活性和可靠性。
企业可以根据实际需求灵活调整资源配置，降低IT成本。

主要的云服务提供商包括亚马逊AWS、微软Azure、谷歌云平台等。
        """.strip()
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/api/v1/documents/json", 
                           json=test_data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()
    
    return response.status_code == 200

def test_search_documents():
    """测试搜索文档接口"""
    print("🔍 测试搜索文档接口...")
    
    # 测试查询
    test_queries = [
        "深度学习和神经网络",
        "云计算服务模式",
        "机器学习算法",
        "基础设施即服务"
    ]
    
    for query in test_queries:
        print(f"查询: {query}")
        params = {"query": query, "limit": 3}
        response = requests.get(f"{BASE_URL}/api/v1/documents/search", params=params)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"找到 {result['total_results']} 个结果:")
            for i, item in enumerate(result['results'], 1):
                print(f"  {i}. 文档ID: {item['post_id']}")
                print(f"     标题: {item['title']}")
                print(f"     相似度: {item['score']:.4f}")
                print(f"     内容片段: {item['content'][:100]}...")
                print()
        else:
            print(f"搜索失败: {response.text}")
        print("-" * 60)

def test_search_documents_json():
    """测试搜索文档接口（JSON形式）"""
    print("🔍 测试搜索文档接口（JSON形式）...")
    
    test_data = {
        "query": "人工智能和机器学习",
        "limit": 5
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{BASE_URL}/api/v1/documents/search", 
                           json=test_data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def main():
    """主测试函数"""
    print("🚀 开始测试向量数据库API接口")
    print("=" * 60)
    
    try:
        # 1. 测试健康检查
        test_health()
        
        # 2. 测试保存文档（表单形式）
        success1 = test_save_document()
        
        # 3. 测试保存文档（JSON形式）
        success2 = test_save_document_json()
        
        # 等待一下让向量生成完成
        if success1 or success2:
            print("⏳ 等待向量生成完成...")
            time.sleep(5)
        
        # 4. 测试搜索功能
        test_search_documents()
        
        # 5. 测试JSON搜索
        test_search_documents_json()
        
        print("✅ 所有测试完成!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保服务正在运行在 http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")

if __name__ == "__main__":
    main()
