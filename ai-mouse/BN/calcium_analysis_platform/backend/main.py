from datetime import datetime
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd

# 导入核心逻辑模块
from src.extraction_logic import run_batch_extraction, extract_calcium_features, get_interactive_data, extract_manual_range
from src.clustering_logic import (
    load_data,
    enhance_preprocess_data,
    cluster_kmeans,
    visualize_clusters_2d,
    visualize_feature_distribution,
    analyze_clusters
)
from src.file_utils import (
    InvalidFile,
    copy_limited,
    resolve_existing_file,
    unique_upload_path,
)
from src.utils import save_plot_as_base64

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="钙信号分析平台 API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vue开发服务器地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 所有运行时文件固定写入后端目录，避免依赖启动命令的工作目录。
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
TEMP_DIR = BASE_DIR / "temp"

for dir_path in [UPLOADS_DIR, RESULTS_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def store_upload(upload: UploadFile, directory: Path) -> Path:
    """Persist one validated spreadsheet below a controlled directory."""

    try:
        destination = unique_upload_path(directory, upload.filename)
        copy_limited(upload.file, destination)
        return destination
    except InvalidFile as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def request_error(exc: Exception, context: str) -> HTTPException:
    """Map expected client errors and hide unexpected internal details."""

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, InvalidFile):
        return HTTPException(status_code=400, detail=str(exc))
    LOGGER.exception("%s failed", context)
    return HTTPException(status_code=500, detail="处理失败，请检查输入格式和服务日志")


@app.get("/")
async def root():
    return {"message": "钙信号分析平台 API"}


@app.post("/api/extraction/preview")
async def preview_extraction(
    file: UploadFile = File(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(12),
    max_duration_frames: int = Form(800),
    min_snr: float = Form(3.5),
    smooth_window: int = Form(31),
    peak_distance_frames: int = Form(24),
    filter_strength: float = Form(1.0),
    neuron_id: str = Form(...)
):
    """预览单个神经元的事件提取结果"""
    temp_file = None
    try:
        temp_file = store_upload(file, TEMP_DIR)
        
        # 读取数据
        df = pd.read_excel(temp_file, sheet_name='dF', header=0)
        
        # 如果neuron_id是'temp'，只返回神经元列表
        if neuron_id == 'temp':
            return {
                "success": True,
                "neuron_columns": df.columns[1:].tolist(),
                "features": [],
                "plot": None
            }
        
        if neuron_id not in df.columns:
            raise HTTPException(status_code=400, detail=f"神经元 {neuron_id} 不存在")
        
        # 设置参数
        params = {
            'min_duration': min_duration_frames,
            'max_duration': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 提取特征并生成可视化
        feature_table, fig, _ = extract_calcium_features(
            df[neuron_id].values, fs=fs, visualize=True, params=params
        )
        
        # 将图表转换为base64
        plot_base64 = save_plot_as_base64(fig)
        
        return {
            "success": True,
            "features": feature_table.to_dict('records') if not feature_table.empty else [],
            "plot": plot_base64,
            "neuron_columns": df.columns[1:].tolist()
        }
        
    except Exception as exc:
        raise request_error(exc, "preview extraction") from exc
    finally:
        if temp_file is not None:
            temp_file.unlink(missing_ok=True)


@app.post("/api/extraction/interactive_data")
async def get_interactive_extraction_data(
    file: UploadFile = File(...),
    neuron_id: str = Form(...)
):
    """获取交互式图表数据"""
    temp_file = None
    try:
        temp_file = store_upload(file, TEMP_DIR)
        
        # 获取交互式数据
        interactive_data = get_interactive_data(str(temp_file), neuron_id)
        
        return {
            "success": True,
            "data": interactive_data
        }
        
    except Exception as exc:
        raise request_error(exc, "interactive extraction data") from exc
    finally:
        if temp_file is not None:
            temp_file.unlink(missing_ok=True)


@app.post("/api/extraction/manual_extract")
async def manual_extraction(
    file: UploadFile = File(...),
    neuron_id: str = Form(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(5),
    max_duration_frames: int = Form(100),
    min_snr: float = Form(2.0),
    smooth_window: int = Form(5),
    peak_distance_frames: int = Form(10),
    filter_strength: float = Form(0.1)
):
    """基于用户选择的时间范围进行手动提取"""
    temp_file = None
    try:
        temp_file = store_upload(file, TEMP_DIR)
        
        # 构建参数字典
        params = {
            'fs': fs,
            'min_duration_frames': min_duration_frames,
            'max_duration_frames': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance_frames': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 执行手动提取
        result = extract_manual_range(str(temp_file), neuron_id, start_time, end_time, params)
        
        return result

    except Exception as exc:
        raise request_error(exc, "manual extraction") from exc
    finally:
        if temp_file is not None:
            temp_file.unlink(missing_ok=True)


@app.post("/api/extraction/batch")
async def batch_extraction(
    files: List[UploadFile] = File(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(12),
    max_duration_frames: int = Form(800),
    min_snr: float = Form(3.5),
    smooth_window: int = Form(31),
    peak_distance_frames: int = Form(24),
    filter_strength: float = Form(1.0)
):
    """批量处理文件进行事件提取"""
    try:
        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        upload_dir = UPLOADS_DIR / timestamp
        upload_dir.mkdir(exist_ok=True)
        
        # 保存上传的文件
        saved_file_paths = []
        for file in files:
            file_path = store_upload(file, upload_dir)
            saved_file_paths.append(str(file_path))
        
        # 设置参数
        params = {
            'min_duration': min_duration_frames,
            'max_duration': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 执行批量提取
        result_path = run_batch_extraction(saved_file_paths, str(RESULTS_DIR), fs=fs, **params)
        
        resolved_result = Path(result_path).resolve() if result_path else None
        if (
            resolved_result is not None
            and resolved_result.is_file()
            and resolved_result.parent == RESULTS_DIR.resolve()
            and resolved_result.suffix.lower() == ".xlsx"
        ):
            return {
                "success": True,
                "result_file": resolved_result.name,
                "message": "批量分析完成"
            }
        raise HTTPException(status_code=500, detail="批量分析未生成有效结果")

    except Exception as exc:
        raise request_error(exc, "batch extraction") from exc


@app.get("/api/results/files")
async def list_result_files():
    """获取结果文件列表"""
    try:
        feature_files = list(RESULTS_DIR.glob("*_features.xlsx"))
        files_info = []
        
        for file_path in feature_files:
            try:
                # 尝试从文件名解析时间戳
                basename = file_path.name
                timestamp_str = basename.split('_features.xlsx')[0].split('_')[-1]
                dt_obj = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
                friendly_name = f"{basename} (创建于: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')})"
            except (ValueError, IndexError):
                friendly_name = basename
            
            files_info.append({
                "filename": basename,
                "friendly_name": friendly_name
            })
        
        return {"files": files_info}
        
    except Exception as exc:
        raise request_error(exc, "result listing") from exc


@app.post("/api/clustering/analyze")
async def clustering_analysis(
    filename: str = Form(...),
    k_value: int = Form(3, ge=2, le=20),
    dim_reduction_method: str = Form("pca")
):
    """执行聚类分析"""
    try:
        try:
            file_path = resolve_existing_file(RESULTS_DIR, filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="文件不存在")
        method = dim_reduction_method.lower()
        if method not in {"pca", "tsne"}:
            raise HTTPException(status_code=400, detail="降维方法仅支持 pca 或 tsne")
        
        # 加载数据
        df = load_data(str(file_path))
        
        # 预处理
        features_scaled, feature_names, df_clean = enhance_preprocess_data(df)
        if k_value > len(df_clean):
            raise HTTPException(status_code=400, detail="聚类数不能大于有效样本数")
        
        # 聚类
        labels = cluster_kmeans(features_scaled, k_value)
        df_clean['cluster'] = labels
        
        # 分析
        cluster_summary = analyze_clusters(df_clean.drop('cluster', axis=1), labels)
        
        # 可视化
        fig_2d = visualize_clusters_2d(features_scaled, labels, feature_names, method=method)
        fig_dist = visualize_feature_distribution(df_clean, labels)
        
        # 转换图表为base64
        plot_2d_base64 = save_plot_as_base64(fig_2d)
        plot_dist_base64 = save_plot_as_base64(fig_dist)
        
        # 保存结果
        output_basename = file_path.stem.removesuffix('_features')
        output_filename = f"{output_basename}_clustered_k{k_value}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.xlsx"
        output_path = RESULTS_DIR / output_filename
        df_clean.to_excel(output_path, index=False)
        
        return {
            "success": True,
            "summary": cluster_summary.to_dict('records'),
            "plot_2d": plot_2d_base64,
            "plot_dist": plot_dist_base64,
            "result_file": output_filename,
            "k_value": k_value,
            "method": method
        }

    except Exception as exc:
        raise request_error(exc, "clustering analysis") from exc


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """下载结果文件"""
    try:
        file_path = resolve_existing_file(RESULTS_DIR, filename)
    except (FileNotFoundError, InvalidFile):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
