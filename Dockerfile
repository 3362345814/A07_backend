# 使用官方Python基础镜像
FROM python:3.11.8-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app
COPY . .

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 配置Django
RUN mkdir -p /app/staticfiles && \
    python manage.py collectstatic --noinput

# 暴露端口
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "180", "A07_backend.wsgi:application"]