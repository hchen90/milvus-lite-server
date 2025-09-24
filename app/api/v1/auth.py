"""
JWT 令牌验证相关的 API 路由（无登录功能）
"""
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Optional

from app.services.auth import get_current_user
from app.core.config import config

logger = logging.getLogger(__name__)

router = APIRouter()


class TokenVerifyResponse(BaseModel):
    """令牌验证响应模型"""
    success: bool = Field(..., description="是否有效")
    message: str = Field(..., description="响应消息")
    username: Optional[str] = Field(None, description="用户名")
    expires_at: Optional[int] = Field(None, description="过期时间戳")


@router.post("/auth/verify",
             response_model=TokenVerifyResponse,
             summary="验证令牌",
             description="验证 JWT 访问令牌的有效性")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """
    验证 JWT 令牌
    
    需要在请求头中提供有效的 Bearer token（当JWT验证启用时）
    """
    try:
        logger.info(f"令牌验证成功: {current_user.get('username', 'unknown')}")
        
        # 如果JWT验证被禁用，返回相应的提示信息
        if not config.JWT_ENABLED:
            return TokenVerifyResponse(
                success=True,
                message="JWT验证已禁用，允许匿名访问",
                username=current_user.get("username"),
                expires_at=current_user.get("token_exp")
            )
        
        return TokenVerifyResponse(
            success=True,
            message="令牌有效",
            username=current_user.get("username"),
            expires_at=current_user.get("token_exp")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"令牌验证过程中发生未预期的错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器内部错误"
        )


@router.get("/auth/profile",
            summary="获取用户信息",
            description="获取当前认证用户的基本信息")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """
    获取用户信息
    
    需要在请求头中提供有效的 Bearer token（当JWT验证启用时）
    """
    try:
        logger.info(f"获取用户信息: {current_user.get('username', 'unknown')}")
        
        # 如果JWT验证被禁用，返回相应的提示信息
        message = "JWT验证已禁用，返回匿名用户信息" if not config.JWT_ENABLED else "用户信息获取成功"
        
        return {
            "success": True,
            "message": message,
            "user": {
                "username": current_user.get("username"),
                "email": current_user.get("email"),
                "full_name": current_user.get("full_name"),
                "token_expires_at": current_user.get("token_exp")
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息过程中发生未预期的错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器内部错误"
        )
