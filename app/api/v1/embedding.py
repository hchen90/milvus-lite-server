import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.api.models import ErrorResponse
from app.services.embedding import get_embedding_from_content, get_embeddings_from_content
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbeddingRequest(BaseModel):
    """单个内容嵌入请求模型"""
    content: str = Field(..., description="要生成嵌入向量的内容", min_length=1)


class EmbeddingResponse(BaseModel):
    """单个内容嵌入响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    content: str = Field(..., description="输入的内容")
    embedding: List[float] = Field(..., description="生成的嵌入向量")
    dimension: int = Field(..., description="向量维度")


class ChunkedEmbeddingResponse(BaseModel):
    """分块内容嵌入响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    content: str = Field(..., description="输入的内容")
    total_chunks: int = Field(..., description="总分块数")
    embeddings: List[Dict[str, Any]] = Field(..., description="分块嵌入结果列表")


@router.post("/embedding",
             response_model=EmbeddingResponse,
             responses={
                 400: {"model": ErrorResponse},
                 500: {"model": ErrorResponse}
             },
             summary="生成单个内容的嵌入向量",
             description="为给定的文本内容生成嵌入向量，适用于短文本或不需要分块的内容")
async def get_single_embedding(request: EmbeddingRequest):
    """
    生成单个内容的嵌入向量
    
    - **content**: 要生成嵌入向量的文本内容
    
    返回单个嵌入向量，适用于短文本处理
    """
    try:
        # 验证输入
        if not request.content or not request.content.strip():
            raise HTTPException(
                status_code=400,
                detail="content不能为空"
            )
        
        content = request.content.strip()
        
        logger.info(f"开始生成单个嵌入向量，内容长度: {len(content)}")
        
        # 调用嵌入服务
        embedding = get_embedding_from_content(content)
        
        if embedding is None or (hasattr(embedding, 'size') and embedding.size == 0) or (isinstance(embedding, list) and len(embedding) == 0):
            raise HTTPException(
                status_code=500,
                detail="无法生成嵌入向量，内容可能为空或模型加载失败"
            )
        
        # 转换为列表格式（如果是numpy数组）
        if hasattr(embedding, 'tolist'):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)
        
        logger.info(f"单个嵌入向量生成成功，维度: {len(embedding_list)}")
        
        return EmbeddingResponse(
            success=True,
            message="嵌入向量生成成功",
            content=content,
            embedding=embedding_list,
            dimension=len(embedding_list)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成单个嵌入向量时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/embeddings",
             response_model=ChunkedEmbeddingResponse,
             responses={
                 400: {"model": ErrorResponse},
                 500: {"model": ErrorResponse}
             },
             summary="生成分块内容的嵌入向量",
             description="为给定的文本内容生成分块嵌入向量，适用于长文本处理")
async def get_chunked_embeddings(request: EmbeddingRequest):
    """
    生成分块内容的嵌入向量
    
    - **content**: 要生成嵌入向量的文本内容
    
    自动将长文本分块并为每个块生成嵌入向量，适用于长文本处理
    """
    try:
        # 验证输入
        if not request.content or not request.content.strip():
            raise HTTPException(
                status_code=400,
                detail="content不能为空"
            )
        
        content = request.content.strip()
        
        logger.info(f"开始生成分块嵌入向量，内容长度: {len(content)}")
        
        # 调用嵌入服务
        embeddings = get_embeddings_from_content(content)
        
        if not embeddings or len(embeddings) == 0:
            raise HTTPException(
                status_code=500,
                detail="无法生成嵌入向量，内容可能为空或模型加载失败"
            )
        
        logger.info(f"分块嵌入向量生成成功，总块数: {len(embeddings)}")
        
        return ChunkedEmbeddingResponse(
            success=True,
            message="分块嵌入向量生成成功",
            content=content,
            total_chunks=len(embeddings),
            embeddings=embeddings
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成分块嵌入向量时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )
