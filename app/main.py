import sys
import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymilvus import MilvusClient

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.core.config import config
from app.services.milvusdb import setup_milvus_collection, create_index
from app.api.v1 import vector_save, vector_search

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储Milvus客户端
milvus_client = None
COLLECTION_NAME = "documents"
COLLECTION_DIMENSION = 384  # paraphrase-multilingual-MiniLM-L12-v2 模型的维度


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    global milvus_client
    
    logger.info("Initializing Milvus vector database...")
    
    try:
        # 初始化Milvus客户端
        milvus_client = MilvusClient(uri=config.MILVUS_PATH)
        logger.info(f"Milvus client initialized with database: {config.MILVUS_PATH}")
        
        # 设置集合
        setup_milvus_collection(
            client=milvus_client,
            collection_name=COLLECTION_NAME,
            collection_dimension=COLLECTION_DIMENSION,
            force=False
        )
        
        # 创建索引
        create_index(
            client=milvus_client,
            collection_name=COLLECTION_NAME
        )
        
        logger.info("Milvus vector database initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# 创建FastAPI应用实例
app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    debug=config.DEBUG,
    lifespan=lifespan
)


def get_milvus_client():
    """获取Milvus客户端实例"""
    global milvus_client
    return milvus_client


# 注册API路由
app.include_router(vector_save.router, prefix="/api/v1", tags=["vector-save"])
app.include_router(vector_search.router, prefix="/api/v1", tags=["vector-search"])


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": f"Welcome to {config.APP_NAME}",
        "version": config.VERSION,
        "server": config.get_server_address()
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "app_name": config.APP_NAME,
        "version": config.VERSION
    }


def main():
    """启动服务器"""
    print(f"Starting {config.APP_NAME} v{config.VERSION}")
    print(f"Server will run on: {config.get_server_address()}")
    print(f"Debug mode: {config.DEBUG}")
    
    # 验证配置
    if not config.validate_config():
        print("Configuration validation failed!")
        return
    
    # 启动服务器
    uvicorn.run(
        "app.main:app",  # 使用绝对模块路径
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
