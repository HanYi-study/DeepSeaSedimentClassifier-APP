"""
全局配置与默认参数
===============
修改此文件即可调整软件默认行为，无需改动核心代码。
"""

# ==================== 默认分类参数 ====================
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 512
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 2
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_RANDOM_SEED = 42

# ==================== 沉积物类型 ====================
# 默认 5 分类。可通过 GUI 自定义类别数和名称。

# 英文名称 (用于图表显示, matplotlib 不支持中文)
SEDIMENT_CLASSES_EN = {
    1: "Calcareous bio-silt",
    2: "Calcareous bio-clay silt",
    3: "Silty sand",
    4: "Medium sand",
    5: "Gravel sand",
}

# 中文名称 (用于日志和导出)
SEDIMENT_CLASSES_CN = {
    1: "钙质生物粉砂",
    2: "钙质生物黏土质粉砂",
    3: "粉砂质砂",
    4: "中砂",
    5: "砾砂",
}

# 实际使用的类别 (默认英文)
SEDIMENT_CLASSES = SEDIMENT_CLASSES_EN

# 分类颜色映射 (RGB)
SEDIMENT_COLORS = {
    1: (255, 255, 150),
    2: (150, 200, 100),
    3: (100, 180, 220),
    4: (220, 150, 80),
    5: (200, 80, 60),
}

# 可自定义更多颜色
SEDIMENT_COLORS_EXTENDED = [
    (255, 255, 150), (150, 200, 100), (100, 180, 220),
    (220, 150, 80), (200, 80, 60), (180, 120, 200),
    (80, 180, 180), (220, 220, 100), (150, 150, 150),
    (100, 220, 100),
]

# ==================== 数据格式 ====================
# TXT 测线数据列定义
# 默认: 经度, 纬度, 反射强度, 序号
TXT_COLUMN_NAMES = ["Longitude", "Latitude", "Intensity", "Sequence"]
TXT_DELIMITER = ","

# ==================== 可视化 ====================
FIGURE_DPI = 100
PROFILE_FIGSIZE = (10, 5)
MAP_FIGSIZE = (8, 6)

# ==================== 导出 ====================
EXPORT_DELIMITER = ","
EXPORT_ENCODING = "utf-8"
