FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝app目录
COPY app/ ./app/

# 启动命令（假设 main.py 是入口）
CMD ["python3", "app/main.py"]
