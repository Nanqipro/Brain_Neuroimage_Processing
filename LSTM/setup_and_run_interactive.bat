@echo off
echo ===== 神经元网络交互式可视化设置 =====
echo.
echo 正在安装必要的库...
pip install pyvis matplotlib networkx numpy scipy scikit-learn
echo.

echo 正在生成交互式神经元网络可视化...
python src/create_interactive_network.py

echo.
echo 如果生成成功，交互式可视化文件将保存在 results/analysis/interactive 目录下。
echo 请使用浏览器打开 interactive_neuron_network.html 文件以查看交互式神经元网络。
echo.

echo 是否自动打开交互式可视化? (Y/N)
set /p open_browser=
if /i "%open_browser%"=="Y" (
    start "" "results/analysis/interactive/interactive_neuron_network.html"
)

pause 