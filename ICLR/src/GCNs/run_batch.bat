@echo off
REM 批量实验运行脚本 (Windows)

echo 开始批量GCN实验...
echo 当前目录: %CD%

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python
    pause
    exit /b 1
)

REM 检查必要文件
if not exist "run.py" (
    echo 错误: 未找到run.py文件，请确保在GCNs目录下运行
    pause
    exit /b 1
)

if not exist "batch_experiments.py" (
    echo 错误: 未找到batch_experiments.py文件
    pause
    exit /b 1
)

REM 创建结果目录
if not exist "batch_results" mkdir batch_results

REM 运行批量实验
echo 开始运行批量实验...
python batch_experiments.py %*

echo 批量实验完成!
echo 结果保存在 batch_results\ 目录中
pause