#!/bin/bash

echo "========================================"
echo "启动大象识别Web应用"
echo "========================================"
echo ""

echo "检查是否已安装Streamlit..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "未检测到Streamlit，正在安装..."
    pip3 install streamlit pandas
fi

echo ""
echo "正在启动Web应用..."
echo "浏览器将自动打开，如未打开请访问: http://localhost:8501"
echo ""
echo "按 Ctrl+C 可停止服务器"
echo ""

streamlit run web_app.py
