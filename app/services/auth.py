"""
JWT 认证服务模块（仅验证，不提供登录）
"""
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import config

logger = logging.getLogger(__name__)

# HTTP Bearer Token 方案
security = HTTPBearer()


class AuthService:
    """认证服务类"""
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """验证访问令牌"""
        try:
            payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing subject",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.PyJWTError as e:
            logger.error(f"JWT validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    获取当前用户信息的依赖函数
    用于 FastAPI 路由的依赖注入
    """
    token = credentials.credentials
    payload = AuthService.verify_token(token)
    
    # 从 payload 中提取用户信息
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing username",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return {
        "username": username,
        "email": payload.get("email"),
        "full_name": payload.get("full_name"),
        "token_exp": payload.get("exp")
    }


def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """
    获取当前用户信息的可选依赖函数
    如果没有提供认证头，返回 None
    """
    if not credentials:
        return None
    
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None