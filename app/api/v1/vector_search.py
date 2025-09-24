import logging
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional

from app.api.models import DocumentSearchRequest, DocumentSearchResponse, SearchResult, ErrorResponse
from app.services.milvusdb import search_data
from app.services.auth import get_current_user
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional

from app.api.models import DocumentSearchRequest, DocumentSearchResponse, SearchResult, ErrorResponse
from app.services.milvusdb import search_data

logger = logging.getLogger(__name__)

router = APIRouter()


def get_milvus_client():
    """获取Milvus客户端"""
    from app.main import get_milvus_client
    return get_milvus_client()


@router.get("/documents/search",
            response_model=DocumentSearchResponse,
            responses={
                400: {"model": ErrorResponse},
                500: {"model": ErrorResponse}
            },
            summary="搜索相似文档",
            description="根据查询文本在向量数据库中搜索相似的文档内容")
async def search_documents(
    query: str = Query(..., description="搜索查询文本", min_length=1),
    limit: int = Query(default=5, description="返回结果数量限制", ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """
    搜索相似文档
    
    - **query**: 搜索查询文本
    - **limit**: 返回结果的数量限制 (1-100, 默认5)
    
    返回与查询文本最相似的文档片段列表，按相似度排序
    """
    try:
        # 验证输入参数
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="查询文本不能为空"
            )
        
        query = query.strip()
        
        logger.info(f"开始搜索文档: query={query[:100]}..., limit={limit}, 用户: {current_user.get('username')}")
        
        # 获取Milvus客户端
        client = get_milvus_client()
        if not client:
            raise HTTPException(
                status_code=500,
                detail="Milvus数据库连接不可用"
            )
        
        # 调用search_data搜索相似文档
        search_results = search_data(
            client=client,
            collection_name="documents",  # 使用固定的集合名称
            text=query,
            limit=limit
        )
        
        # 转换搜索结果格式
        results = []
        for result in search_results:
            results.append(SearchResult(
                id=result.get("id"),
                post_id=result.get("post_id", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                distance=result.get("distance", 0.0)
            ))
        
        logger.info(f"搜索完成: query={query[:50]}..., found={len(results)} results")
        
        return DocumentSearchResponse(
            success=True,
            message=f"找到 {len(results)} 个相关结果",
            query=query,
            total_results=len(results),
            results=results
        )
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"搜索文档时发生未预期的错误: query={query}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/documents/search",
             response_model=DocumentSearchResponse,
             responses={
                 400: {"model": ErrorResponse},
                 500: {"model": ErrorResponse}
             },
             summary="通过JSON搜索相似文档",
             description="通过JSON格式的请求体搜索相似文档")
async def search_documents_json(request: DocumentSearchRequest, current_user: dict = Depends(get_current_user)):
    """
    通过JSON格式搜索相似文档
    
    这是一个备用接口，支持JSON格式的请求体
    """
    try:
        logger.info(f"开始搜索文档(JSON): query={request.query[:100]}..., limit={request.limit}, 用户: {current_user.get('username')}")
        
        # 获取Milvus客户端
        client = get_milvus_client()
        if not client:
            raise HTTPException(
                status_code=500,
                detail="Milvus数据库连接不可用"
            )
        
        # 调用search_data搜索相似文档
        search_results = search_data(
            client=client,
            collection_name="documents",
            text=request.query,
            limit=request.limit
        )
        
        # 转换搜索结果格式
        results = []
        for result in search_results:
            results.append(SearchResult(
                id=result.get("id"),
                post_id=result.get("post_id", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                distance=result.get("distance", 0.0)
            ))
        
        logger.info(f"搜索完成(JSON): query={request.query[:50]}..., found={len(results)} results")
        
        return DocumentSearchResponse(
            success=True,
            message=f"找到 {len(results)} 个相关结果",
            query=request.query,
            total_results=len(results),
            results=results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索文档时发生未预期的错误(JSON): query={request.query}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )