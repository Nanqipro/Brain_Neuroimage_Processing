import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import json
from PIL import Image
import time

# --- Streamlit 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(page_title="LSTM 神经元分析", layout="wide")

# --- 路径设置和模块导入 ---

# 假设脚本在 LSTM/src/ 目录下运行
# 确定 torpedo 模块的路径
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TORPEDO_DIR = os.path.join(SRC_DIR, 'torpedo')

# 将 torpedo 目录添加到 Python 路径
if TORPEDO_DIR not in sys.path:
    sys.path.insert(0, TORPEDO_DIR)

# 导入配置 (现在应该可以找到了)
try:
    from config import LSTMConfig
except ImportError as e:
    # 错误处理现在可以在 set_page_config 之后安全地进行
    st.error(f"无法导入配置模块: {e}\n请确保 torpedo 目录在 Python 路径中，并且 config.py 存在于 {TORPEDO_DIR}")
    st.stop() # 停止脚本执行

# --- 配置加载 ---
@st.cache_resource # 缓存配置对象
def load_config():
    try:
        return LSTMConfig()
    except Exception as e:
        st.error(f"加载 LSTMConfig 时出错: {e}")
        return None

config = load_config()
if config is None:
    st.stop() # 如果加载失败则停止

# --- 助手函数 ---
def run_script(script_path: str):
    """运行指定的 Python 脚本或模块并显示输出。"""
    
    command = []
    module_name = None
    display_name = script_path # 默认显示名称

    # 检查路径是否指向 torpedo 包内的模块
    # train.py 仍然使用模块化运行
    if script_path == "torpedo/train.py":
        module_name = "torpedo.train"
        command = [sys.executable, "-m", module_name]
        full_script_path_check = os.path.join(SRC_DIR, script_path)
        display_name = module_name # 更新显示名称为模块名
    elif script_path == "evaluate_lstm.py": # 将 evaluate_lstm.py 改回普通脚本运行
        command = [sys.executable, script_path] # 直接运行脚本
        full_script_path_check = os.path.join(SRC_DIR, script_path)
        display_name = script_path # 显示脚本名
        st.info("运行 evaluate_lstm.py 作为普通脚本。") # 添加提示
    else:
        # 其他脚本（如果需要）
        st.warning(f"脚本 {script_path} 未特别处理，尝试作为普通脚本运行。")
        command = [sys.executable, script_path]
        full_script_path_check = os.path.join(SRC_DIR, script_path)

    # 检查文件是否存在
    if not os.path.exists(full_script_path_check):
        st.error(f"脚本文件未找到: {full_script_path_check}")
        return
        
    st.info(f"正在运行: {' '.join(command)} ...") # 显示完整命令
    log_placeholder = st.empty()
    log_content = ""
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        cwd=SRC_DIR # 保持在 src 目录运行
    )

    # 实时读取输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            log_content += output
            # 每隔一小段时间更新一次，避免过于频繁刷新 Streamlit
            log_placeholder.code(log_content, language='bash') 
            time.sleep(0.05) # 短暂暂停
    
    # 确保最终输出完全显示
    log_placeholder.code(log_content, language='bash')
    
    # 获取最终返回码
    return_code = process.poll()
    if return_code == 0:
        st.success(f"运行 {display_name} 成功完成。")
    else:
        st.error(f"运行 {display_name} 执行出错，返回码: {return_code}")

# --- Streamlit 应用程序主体 ---

# 标题现在可以放在这里
st.title("🧠 LSTM 神经元活动分析平台")

# --- 侧边栏导航 ---
st.sidebar.title("导航")
# 添加 "📄 查看报告" 选项
page = st.sidebar.radio("选择页面", ["🏠 主页", "⚙️ 模型训练", "📊 模型评估", "📈 查看结果", "📄 查看报告"])

st.sidebar.markdown("--- ")
st.sidebar.info(f"数据标识符: `{config.data_identifier}`")

# --- 页面内容 ---
if page == "🏠 主页":
    st.header("欢迎使用 LSTM 神经元分析平台")
    st.markdown("""
    此平台用于训练、评估和可视化基于 LSTM 的模型，该模型旨在分析神经元活动数据与行为标签之间的关系。

    **功能:**
    *   **模型训练:** 使用 `torpedo/train.py` 脚本启动 LSTM 模型的训练过程。
    *   **模型评估:** 使用 `torpedo/evaluate_lstm.py` 脚本在测试集上评估已训练的模型。
    *   **查看结果:** 可视化训练过程中的指标、测试集的混淆矩阵以及详细的评估报告。
    *   **查看报告:** 查看自动生成的 Markdown 格式的总结报告。

    请使用侧边栏导航到不同功能页面。
    """)
    st.markdown("--- ")
    st.subheader("当前配置概览 (`config.py`)")
    # 显示部分关键配置项
    config_dict = {
        "Data Paths": {
            "Data File": config.data_file,
            "Model Directory": config.model_dir,
            "Log Directory": config.log_dir,
            "Plot Directory": config.plot_dir,
            "Model Save Path": config.model_path,
            "Scaler Save Path": config.scaler_path
        },
        "Model Hyperparameters": {
            "Sequence Length": config.sequence_length,
            "Hidden Size": config.hidden_size,
            "Num Layers": config.num_layers,
            "Latent Dim": config.latent_dim,
            "Num Heads": config.num_heads,
            "Dropout": config.dropout
        },
        "Training Parameters": {
            "Batch Size": config.batch_size,
            "Num Epochs": config.num_epochs,
            "Learning Rate": config.learning_rate,
            "Weight Decay": config.weight_decay,
            "Reconstruction Loss Weight": config.reconstruction_loss_weight,
            "Gradient Clip Norm": config.gradient_clip_norm,
            "Early Stopping Enabled": config.early_stopping_enabled,
            "Early Stopping Patience": config.early_stopping_patience
        }
    }
    st.json(config_dict)

elif page == "⚙️ 模型训练":
    st.header("模型训练")
    st.markdown("点击下面的按钮开始模型训练过程。训练日志将实时显示。")
    st.markdown("训练将使用 `torpedo/config.py` 中定义的参数。")
    
    st.subheader("启动训练")
    if st.button("🚀 开始训练", key="train_button"):
        # 提供相对于 SRC_DIR 的正确路径
        run_script("torpedo/train.py")
    
elif page == "📊 模型评估":
    st.header("模型评估")
    st.markdown("点击下面的按钮在测试集上评估已训练的模型。评估日志将显示。")
    st.markdown("评估将加载保存在 `config.model_path` 的模型和 `config.scaler_path` 的 Scaler。")
    
    st.subheader("启动评估")
    if st.button("🔍 开始评估", key="evaluate_button"):
        # 检查模型和 scaler 文件是否存在
        if not os.path.exists(config.model_path):
            st.warning(f"模型文件未找到: {config.model_path}。请先完成训练。")
        elif not os.path.exists(config.scaler_path):
            st.warning(f"Scaler 文件未找到: {config.scaler_path}。请先完成训练。")
        else:
            run_script("evaluate_lstm.py")

elif page == "📈 查看结果":
    st.header("查看结果")
    st.markdown(f"显示与数据标识符 `{config.data_identifier}` 相关的训练和评估结果。")

    # 检查并显示训练指标图
    st.subheader("训练过程指标")
    # 添加说明文字
    st.markdown("**注意：** 以下训练曲线图（损失、准确率）来自 K-Fold 交叉验证阶段（可能显示最后一个 Fold 的结果）。最终评估的模型是在 K-Fold 之后使用完整训练集重新训练的。", unsafe_allow_html=True)
    
    combined_plot_path = os.path.join(config.plot_dir, f"training_metrics_{config.data_identifier}.png")
    accuracy_plot_path = config.accuracy_plot
    loss_plot_path = config.loss_plot

    if os.path.exists(combined_plot_path):
        try:
            image = Image.open(combined_plot_path)
            st.image(image, caption="训练指标 (损失、准确率、重构损失)", use_container_width=True)
        except Exception as e:
            st.warning(f"无法加载合并的训练指标图: {e}")
    else:
        st.info("合并的训练指标图尚未生成。请先运行模型训练。")
        # 尝试显示单独的图
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(accuracy_plot_path):
                try:
                    image = Image.open(accuracy_plot_path)
                    st.image(image, caption="训练/验证准确率", use_container_width=True)
                except Exception as e:
                    st.warning(f"无法加载准确率图: {e}")
            else:
                st.info("准确率图不存在。")
        with col2:
            if os.path.exists(loss_plot_path):
                try:
                    image = Image.open(loss_plot_path)
                    st.image(image, caption="训练/验证/重构损失", use_container_width=True)
                except Exception as e:
                    st.warning(f"无法加载损失图: {e}")
            else:
                st.info("损失图不存在。")
                
    st.markdown("--- ")
    
    # 检查并显示混淆矩阵
    st.subheader("测试集混淆矩阵")
    cm_plot_path = config.confusion_matrix_plot
    if os.path.exists(cm_plot_path):
        try:
            image = Image.open(cm_plot_path)
            st.image(image, caption="测试集混淆矩阵", use_container_width=True)
        except Exception as e:
            st.warning(f"无法加载混淆矩阵图: {e}")
    else:
        st.info("混淆矩阵图尚未生成。请先运行模型评估。")
        
    st.markdown("--- ")
    
    # 检查并显示评估结果 JSON
    st.subheader("测试集评估指标")
    results_path = config.eval_results_json
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results_data = json.load(f)
            
            # 显示关键指标
            st.metric("测试集准确率", f"{results_data.get('test_set_accuracy', 0)*100:.2f}%")
            st.metric("测试集 AUC-ROC (Macro Avg)", f"{results_data.get('test_set_auc_roc_ovr_macro', 'N/A'):.4f}")
            st.metric("测试集平均精度 (Macro Avg)", f"{results_data.get('test_set_average_precision_macro', 'N/A'):.4f}")
            
            # 显示分类报告 (使用 DataFrame)
            report_dict = results_data.get('test_set_classification_report')
            if report_dict:
                st.text("分类报告:")
                # 转换报告字典为 DataFrame
                df_report = pd.DataFrame(report_dict).transpose()
                # 格式化浮点数列
                float_cols = df_report.select_dtypes(include=['float']).columns
                format_dict = {col: '{:.4f}' for col in float_cols if col != 'support'}
                format_dict['support'] = '{:.0f}'
                st.dataframe(df_report.style.format(format_dict))
            
            # 显示原始 JSON 数据供参考
            with st.expander("查看完整的评估结果 JSON 数据"): 
                st.json(results_data)
                
        except Exception as e:
            st.warning(f"无法加载或解析评估结果 JSON 文件: {e}")
    else:
        st.info("评估结果 JSON 文件尚未生成。请先运行模型评估。")

# --- 新增报告页面 ---
elif page == "📄 查看报告":
    st.header("查看分析报告")
    st.markdown(f"显示为数据标识符 `{config.data_identifier}` 生成的 Markdown 总结报告。")
    
    # 动态导入 report_utils 来获取路径 (避免在顶部导入导致循环依赖或错误)
    try:
        from torpedo.report_utils import get_report_path
        report_path = get_report_path(config)
        
        if os.path.exists(report_path):
            st.success(f"找到报告文件: `{report_path}`")
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                st.markdown(report_content) # 直接渲染 Markdown
            except Exception as e:
                st.error(f"读取或显示报告时出错: {e}")
        else:
            st.warning(f"报告文件尚未生成: `{report_path}`")
            st.info("请先运行训练和评估流程来生成报告。")
            
    except ImportError:
        st.error("无法导入 report_utils。请确保文件存在且路径正确。")
    except Exception as e:
        st.error(f"获取报告路径或处理报告时出错: {e}")

st.sidebar.markdown("--- ")
st.sidebar.markdown("Torpedo LSTM @ Streamlit") 