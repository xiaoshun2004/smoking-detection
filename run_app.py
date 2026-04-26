"""
启动吸烟检测系统 Streamlit 界面
用法：在 smoking-detection 目录下运行
    set PYTHONUTF8=1 && python run_app.py
"""
import sys
import os
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

project_root = Path(__file__).resolve().parent
app_file = project_root / "app" / "app.py"

os.chdir(str(project_root))
sys.argv = ["streamlit", "run", str(app_file),
            "--server.headless", "true",
            "--server.port", "8501"]

from streamlit.web.cli import main
main()
