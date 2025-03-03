FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y graphviz

# 复制项目文件
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 暴露端口（Railway通常使用 $PORT）
EXPOSE 8000

CMD ["streamlit", "run", "bkt_streamlit_pyvis.py", "--server.port=8000", "--server.address=0.0.0.0"]
