import logging
from fastapi import APIRouter, HTTPException, Form, Depends
from typing import Annotated

from app.api.models import DocumentSaveRequest, DocumentSaveResponse, ErrorResponse
from app.services.milvusdb import insert_data
from app.services.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


def get_milvus_client():
    """获取Milvus客户端"""
    from app.main import get_milvus_client
    return get_milvus_client()


@router.post("/documents", 
             response_model=DocumentSaveResponse,
             responses={
                 400: {"model": ErrorResponse},
                 500: {"model": ErrorResponse}
             },
             summary="保存文档到向量数据库",
             description="通过POST表单提交文档数据，支持多行文档内容，并将其转换为向量存储到Milvus数据库中")
async def save_document(
    post_id: Annotated[str, Form(description="文档唯一标识符")],
    title: Annotated[str, Form(description="文档标题")],
    content: Annotated[str, Form(description="文档内容，支持多行文本")],
    current_user: dict = Depends(get_current_user)
):
    """
    保存文档到向量数据库
    
    - **post_id**: 文档的唯一标识符
    - **title**: 文档标题
    - **content**: 文档内容，可以是多行文档数据
    """
    try:
        # 验证输入参数
        if not post_id or not post_id.strip():
            raise HTTPException(
                status_code=400,
                detail="post_id不能为空"
            )
        
        if not title or not title.strip():
            raise HTTPException(
                status_code=400,
                detail="title不能为空"
            )
        
        if not content or not content.strip():
            raise HTTPException(
                status_code=400,
                detail="content不能为空"
            )
        
        # 清理输入数据
        post_id = post_id.strip()
        title = title.strip()
        content = content.strip()
        
        # 验证长度限制
        if len(post_id) > 100:
            raise HTTPException(
                status_code=400,
                detail="post_id长度不能超过100个字符"
            )
        
        if len(title) > 200:
            raise HTTPException(
                status_code=400,
                detail="title长度不能超过200个字符"
            )
        
        logger.info(f"开始保存文档: post_id={post_id}, title={title}, content_length={len(content)}, 用户: {current_user.get('username')}")
        
        # 获取Milvus客户端
        client = get_milvus_client()
        if not client:
            raise HTTPException(
                status_code=500,
                detail="Milvus数据库连接不可用"
            )
        
        # 调用insert_data保存文档
        success = insert_data(
            client=client,
            collection_name="documents",  # 使用固定的集合名称
            post_id=post_id,
            post_title=title,
            post=content
        )
        
        if success:
            logger.info(f"文档保存成功: post_id={post_id}")
            return DocumentSaveResponse(
                success=True,
                message="文档保存成功",
                post_id=post_id
            )
        else:
            logger.error(f"文档保存失败: post_id={post_id}")
            raise HTTPException(
                status_code=500,
                detail="文档保存失败，请检查日志获取详细信息"
            )
    
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"保存文档时发生未预期的错误: post_id={post_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/documents/json",
             response_model=DocumentSaveResponse,
             responses={
                 400: {"model": ErrorResponse},
                 500: {"model": ErrorResponse}
             },
             summary="通过JSON保存文档到向量数据库",
             description="通过JSON格式提交文档数据")
async def save_document_json(request: DocumentSaveRequest, current_user: dict = Depends(get_current_user)):
    """
    通过JSON格式保存文档到向量数据库
    
    这是一个备用接口，支持JSON格式的请求体
    """
    try:
        logger.info(f"开始保存文档(JSON): post_id={request.post_id}, title={request.title}, content_length={len(request.content)}, 用户: {current_user.get('username')}")
        
        # 获取Milvus客户端
        client = get_milvus_client()
        if not client:
            raise HTTPException(
                status_code=500,
                detail="Milvus数据库连接不可用"
            )
        
        # 调用insert_data保存文档
        success = insert_data(
            client=client,
            collection_name="documents",
            post_id=request.post_id,
            post_title=request.title,
            post=request.content
        )
        
        if success:
            logger.info(f"文档保存成功(JSON): post_id={request.post_id}")
            return DocumentSaveResponse(
                success=True,
                message="文档保存成功",
                post_id=request.post_id
            )
        else:
            logger.error(f"文档保存失败(JSON): post_id={request.post_id}")
            raise HTTPException(
                status_code=500,
                detail="文档保存失败，请检查日志获取详细信息"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存文档时发生未预期的错误(JSON): post_id={request.post_id}, error={str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )