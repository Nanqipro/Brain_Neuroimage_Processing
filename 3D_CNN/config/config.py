from pathlib import Path

class Config:
    # 项目根目录
    ROOT_DIR = Path(__file__).parent.parent

    # 数据相关路径
    DATA_DIR = ROOT_DIR / "datasets"
    DATASET_DIR = DATA_DIR / "dataset"
    OUTPUT_DIR = DATA_DIR / "output"
    MODEL_DIR = DATA_DIR / "model"

    # 数据文件
    X_PATH = DATASET_DIR / "X_3d.npy"
    Y_PATH = DATASET_DIR / "y_labels.npy"
    FRAMES_DIR = OUTPUT_DIR / "frames"
    LABELS_CSV = OUTPUT_DIR / "labels6.csv"
    MODEL_PATH = MODEL_DIR / "best_cnn3d_model.pth"

    # 训练参数
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    INPUT_HEIGHT = 256
    INPUT_WIDTH = 256
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 10

    # 可视化相关路径
    VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

    # 确保必要的目录存在
    @classmethod
    def setup(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True) 

    # 可视化设置
    USE_ENGLISH_LABELS = True  # 设置为 True 使用英文标签 