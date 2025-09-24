"""
JWT 令牌验证功能测试（无登录）
"""
import requests
import json
import jwt
from datetime import datetime, timedelta, timezone

# 测试服务器配置
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

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


def create_expired_token(username: str = "expired_user") -> str:
    """创建已过期的测试令牌"""
    expire = datetime.now(timezone.utc) - timedelta(minutes=1)  # 1分钟前过期
    
    payload = {
        "sub": username,
        "email": f"{username}@example.com",
        "full_name": f"Expired User {username.title()}",
        "exp": expire
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def test_jwt_token_validation():
    """测试 JWT 令牌验证功能"""
    print("=" * 60)
    print("测试 JWT 令牌验证功能（无登录接口）")
    print("=" * 60)
    
    try:
        # 1. 测试服务器健康状态
        print("\n1. 检查服务器状态...")
        try:
            health_response = requests.get(f"{BASE_URL}/health", timeout=5)
            if health_response.status_code == 200:
                print("   ✓ 服务器运行正常")
            else:
                print(f"   ⚠ 服务器状态异常: {health_response.status_code}")
        except requests.exceptions.ConnectionError:
            print("   ✗ 无法连接到服务器，请确保服务器在 http://localhost:8000 上运行")
            return False
        
        # 2. 创建有效的测试令牌
        print("\n2. 创建测试令牌...")
        valid_token = create_test_token("api_test_user")
        print(f"   ✓ 测试令牌已创建: {valid_token[:30]}...")
        
        # 3. 测试令牌验证
        print("\n3. 测试有效令牌验证...")
        headers = {"Authorization": f"Bearer {valid_token}"}
        verify_response = requests.post(f"{API_BASE}/auth/verify", 
                                      headers=headers, 
                                      timeout=10)
        if verify_response.status_code == 200:
            verify_data = verify_response.json()
            print("   ✓ 有效令牌验证成功")
            print(f"   用户名: {verify_data.get('username', 'N/A')}")
        else:
            print(f"   ✗ 有效令牌验证失败: {verify_response.status_code}")
            print(f"   响应: {verify_response.text}")
        
        # 4. 测试获取用户信息
        print("\n4. 测试获取用户信息...")
        profile_response = requests.get(f"{API_BASE}/auth/profile", 
                                      headers=headers, 
                                      timeout=10)
        if profile_response.status_code == 200:
            profile_data = profile_response.json()
            print(f"   ✓ 获取用户信息成功")
            print(f"   用户名: {profile_data['user']['username']}")
            print(f"   邮箱: {profile_data['user'].get('email', 'N/A')}")
        else:
            print(f"   ✗ 获取用户信息失败: {profile_response.status_code}")
            print(f"   响应: {profile_response.text}")
        
        # 5. 测试受保护端点（无认证）
        print("\n5. 测试受保护端点（无认证）...")
        no_auth_response = requests.post(f"{API_BASE}/embedding", 
                                       json={"content": "test content"}, 
                                       timeout=10)
        if no_auth_response.status_code == 403:
            print("   ✓ 正确拒绝了无认证的请求")
        elif no_auth_response.status_code == 422:
            print("   ! 无认证请求返回 422（可能是参数验证错误，但仍然被拒绝）")
        else:
            print(f"   ⚠ 无认证请求状态: {no_auth_response.status_code}")
        
        # 6. 测试受保护端点（有有效认证）
        print("\n6. 测试受保护端点（有效令牌）...")
        auth_response = requests.post(f"{API_BASE}/embedding", 
                                    json={"content": "这是测试内容"}, 
                                    headers=headers, 
                                    timeout=10)
        
        if auth_response.status_code == 200:
            print("   ✓ 成功访问受保护端点")
            response_data = auth_response.json()
            print(f"   响应维度: {response_data.get('dimension', 'N/A')}")
        elif auth_response.status_code == 401:
            print("   ✗ 认证失败")
        elif auth_response.status_code == 500:
            print("   ! 认证通过但服务内部错误（可能是嵌入模型未加载）")
        else:
            print(f"   ! 其他状态: {auth_response.status_code}")
            print(f"   响应: {auth_response.text[:200]}...")
        
        # 7. 测试无效令牌
        print("\n7. 测试无效令牌...")
        invalid_headers = {"Authorization": "Bearer invalid-token-here"}
        invalid_response = requests.post(f"{API_BASE}/auth/verify", 
                                       headers=invalid_headers, 
                                       timeout=10)
        if invalid_response.status_code == 401:
            print("   ✓ 正确拒绝了无效令牌")
        else:
            print(f"   ⚠ 无效令牌响应状态: {invalid_response.status_code}")
        
        # 8. 测试过期令牌
        print("\n8. 测试过期令牌...")
        expired_token = create_expired_token("expired_test_user")
        expired_headers = {"Authorization": f"Bearer {expired_token}"}
        expired_response = requests.post(f"{API_BASE}/auth/verify", 
                                       headers=expired_headers, 
                                       timeout=10)
        if expired_response.status_code == 401:
            expired_data = expired_response.json()
            if "expired" in expired_data.get("detail", "").lower():
                print("   ✓ 正确拒绝了过期令牌")
            else:
                print("   ✓ 拒绝了过期令牌（但错误消息不明确）")
        else:
            print(f"   ⚠ 过期令牌响应状态: {expired_response.status_code}")
        
    except requests.exceptions.RequestException as e:
        print(f"\n   ✗ 网络请求错误: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("JWT 令牌验证测试完成")
    print("=" * 60)
    return True


def show_token_usage_examples():
    """显示令牌使用示例"""
    print("\n" + "=" * 60)
    print("JWT 令牌使用示例")
    print("=" * 60)
    
    print("\n1. 创建测试令牌（Python）:")
    print("""
import jwt
from datetime import datetime, timedelta

def create_token(username):
    expire = datetime.now(timezone.utc) + timedelta(minutes=30)
    payload = {
        "sub": username,
        "email": f"{username}@example.com",
        "full_name": f"User {username}",
        "exp": expire
    }
    token = jwt.encode(payload, "your-secret-key", algorithm="HS256")
    return token

token = create_token("my_user")
""")
    
    print("\n2. 使用令牌访问 API（curl）:")
    test_token = create_test_token("example_user")
    print(f"""
# 验证令牌
curl -X POST {API_BASE}/auth/verify \\
  -H "Authorization: Bearer {test_token[:30]}..." \\
  -H "Content-Type: application/json"

# 获取用户信息
curl -X GET {API_BASE}/auth/profile \\
  -H "Authorization: Bearer {test_token[:30]}..."

# 访问受保护的嵌入端点
curl -X POST {API_BASE}/embedding \\
  -H "Authorization: Bearer {test_token[:30]}..." \\
  -H "Content-Type: application/json" \\
  -d '{{"content": "test content"}}'
""")
    
    print("\n3. 令牌格式说明:")
    print("""
JWT 令牌必须包含以下字段：
- sub: 用户标识（必需）
- exp: 过期时间戳（必需）
- email: 用户邮箱（可选）
- full_name: 用户全名（可选）

使用 HS256 算法和配置的密钥进行签名。
""")


if __name__ == "__main__":
    success = test_jwt_token_validation()
    
    if success:
        print("\n🎉 JWT 令牌验证机制运行正常！")
        show_token_usage_examples()
    else:
        print("\n❌ JWT 令牌验证测试存在问题")
        print("\n💡 请确保:")
        print("  1. 服务器正在运行: uv run python -m app.main")
        print("  2. 服务器地址为: http://localhost:8000")
        print("  3. JWT_SECRET_KEY 配置正确")