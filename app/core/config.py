import os
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    """应用配置类"""
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # 应用配置
    APP_NAME: str = os.getenv("APP_NAME", "milvus-lite-server")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    VERSION: str = os.getenv("VERSION", "1.0.0")

    # Milvus 配置
    MILVUS_PATH: str = os.getenv("MILVUS_PATH", f"{APP_NAME}.db")
    
    # JWT 配置
    JWT_ENABLED: bool = os.getenv("JWT_ENABLED", "True").lower() in ("true", "1", "yes")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    
    @classmethod
    def get_server_address(cls) -> str:
        """获取服务器完整地址"""
        return f"http://{cls.HOST}:{cls.PORT}"
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置是否有效"""
        try:
            # 验证端口范围
            if not (1 <= cls.PORT <= 65535):
                raise ValueError(f"Invalid PORT: {cls.PORT}")
            return True
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False


# 全局配置实例
config = Config()