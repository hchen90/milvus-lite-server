from pydantic import BaseModel, Field
from typing import List, Optional


class DocumentSaveRequest(BaseModel):
    """保存文档的请求模型"""
    post_id: str = Field(..., description="文档ID", min_length=1, max_length=100)
    title: str = Field(..., description="文档标题", min_length=1, max_length=200)
    content: str = Field(..., description="文档内容", min_length=1)


class DocumentSaveResponse(BaseModel):
    """保存文档的响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    post_id: str = Field(..., description="文档ID")


class DocumentSearchRequest(BaseModel):
    """搜索文档的请求模型"""
    query: str = Field(..., description="搜索查询文本", min_length=1)
    limit: int = Field(default=5, description="返回结果数量限制", ge=1, le=100)


class SearchResult(BaseModel):
    """搜索结果项"""
    id: Optional[int] = Field(None, description="Milvus记录ID")
    post_id: str = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="匹配的文档内容片段")
    score: float = Field(..., description="相似度分数")
    distance: float = Field(..., description="向量距离")


class DocumentSearchResponse(BaseModel):
    """搜索文档的响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    query: str = Field(..., description="查询文本")
    total_results: int = Field(..., description="结果总数")
    results: List[SearchResult] = Field(..., description="搜索结果列表")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_type: Optional[str] = Field(None, description="错误类型")
