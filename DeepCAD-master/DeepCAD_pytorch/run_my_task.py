import os
import sys
from pathlib import Path

# ================= 配置区域 (针对 RTX 4060 8G 优化) =================
# 项目名称 (对应 datasets 下的文件夹名)
PROJECT_NAME = 'my_experiment'

# 显卡编号
GPU_ID = '0,1,2,3'

BATCH_SIZE = '4'

# 显存优化参数 (核心)
# 默认是 128，但 8G 显存跑 3D 卷积容易崩，改成 64 非常安全且快
IMG_H = '64'   
IMG_W = '64'

# 时间维度 (一次读取多少帧进行训练)
IMG_S = '150'

# 切片间隔 (步长)
GAP_H = '32'   # 建议是 IMG_H 的一半
GAP_W = '32'
GAP_S = '60'

# 训练轮数 (15轮通常足够去除雪花噪点)
EPOCHS = '15'

TEST_SAVE_ALL_PTH = False
TEST_PTH_NAME = ''
# ===================================================================

ROOT_DIR = Path(__file__).resolve().parent
DATASETS_PATH = ROOT_DIR / "datasets"
PTH_PATH = ROOT_DIR / "pth"


def _python() -> str:
    return sys.executable


def _ensure_dataset_ready() -> None:
    project_dir = DATASETS_PATH / PROJECT_NAME
    if not project_dir.exists():
        raise FileNotFoundError(f"找不到数据目录: {project_dir}")
    tif_files = sorted(list(project_dir.glob("*.tif")) + list(project_dir.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"数据目录下未找到 .tif/.tiff 文件: {project_dir}")


def _get_latest_model_dir() -> Path:
    if not PTH_PATH.exists():
        raise FileNotFoundError(f"找不到模型目录: {PTH_PATH}")
    candidates = [
        p for p in PTH_PATH.iterdir()
        if p.is_dir() and p.name.startswith(f"{PROJECT_NAME}_")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"未找到训练输出的模型文件夹（期望位于 {PTH_PATH}/ 下，且以 {PROJECT_NAME}_ 开头）"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_training():
    print(f"🚀 [阶段 1/2] 开始训练 DeepCAD 模型: {PROJECT_NAME} ...")
    _ensure_dataset_ready()
    cmd = (
        f"{_python()} train.py "
        f"--datasets_path datasets "
        f"--datasets_folder {PROJECT_NAME} "
        f"--img_h {IMG_H} --img_w {IMG_W} --img_s {IMG_S} "
        f"--gap_h {GAP_H} --gap_w {GAP_W} --gap_s {GAP_S} "
        f"--n_epochs {EPOCHS} "
        f"--GPU {GPU_ID} "
        f"--batch_size {BATCH_SIZE} "
        f"--train_datasets_size 4000 " # 提取多少个样本用于训练
        f"--select_img_num 10000"       # 限制读取的最大帧数
    )
    exit_code = os.system(cmd)
    if exit_code != 0:
        print("❌ 训练出错！请检查上方报错信息 (常见原因: 显存不足 或 找不到datasets文件夹)")
        exit(1)
    print("✅ 训练完成！模型已保存。")

def run_inference():
    print(f"🚀 [阶段 2/2] 开始使用模型去噪 (推理) ...")
    _ensure_dataset_ready()
    latest_model_dir = _get_latest_model_dir()
    extra_args = ""
    if TEST_SAVE_ALL_PTH:
        extra_args += " --save_all_pth"
    if TEST_PTH_NAME:
        extra_args += f" --pth_name {TEST_PTH_NAME}"
    # 推理时使用同样的切片大小，防止 OOM
    cmd = (
        f"{_python()} test.py "
        f"--pth_path pth "
        f"--denoise_model {latest_model_dir.name} "
        f"--datasets_path datasets "
        f"--datasets_folder {PROJECT_NAME} "
        f"--img_h {IMG_H} --img_w {IMG_W} --img_s {IMG_S} "
        f"--gap_h {GAP_H} --gap_w {GAP_W} --gap_s {GAP_S} "
        f"--GPU {GPU_ID} "
        f"--batch_size {BATCH_SIZE} "
        f"--test_datasize 10000 "
        f"{extra_args}"
    )
    os.system(cmd)
    print(f"✅ 全部完成！去噪结果请在 results/{PROJECT_NAME} 中查看。")

if __name__ == "__main__":
    # run_training()
    run_inference()
