<p align="center">
  <img src="icon.ico" width="96" alt="DSSC Logo">
</p>

<h1 align="center">DeepSea Sediment Classifier (DSSC)</h1>

<p align="center">
  <strong>深海底质智能分类系统</strong><br>
  基于 MSC-Transformer 深度学习模型的海底沉积物自动分类与空间制图软件
</p>

<p align="center">
  <a href="#-快速开始"><img src="https://img.shields.io/badge/Windows-可执行文件-blue?logo=windows" alt="Windows exe"></a>
  <a href="#-开发环境搭建"><img src="https://img.shields.io/badge/Python-3.9+-green?logo=python" alt="Python"></a>
  <a href="#-引用"><img src="https://img.shields.io/badge/论文-JMSE%202023-orange?logo=readthedocs" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-学术研究-lightgrey" alt="License"></a>
</p>

---

## 📖 目录

- [项目简介](#-项目简介)
- [核心功能](#-核心功能)
- [快速开始](#-快速开始)
  - [方式一：直接使用 exe（推荐普通用户）](#方式一直接使用-exe推荐普通用户)
  - [方式二：从源码运行（推荐开发者）](#方式二从源码运行推荐开发者)
- [数据格式说明](#-数据格式说明)
- [用户操作指南](#-用户操作指南)
- [沉积物分类体系](#-沉积物分类体系)
- [模型架构](#-模型架构)
- [项目结构](#-项目结构)
- [开发环境搭建](#-开发环境搭建)
- [开发指南](#-开发指南)
- [打包为独立 exe](#-打包为独立-exe)
- [常见问题](#-常见问题)
- [引用](#-引用)
- [许可证](#-许可证)
- [联系方式](#-联系方式)

---

## 📌 项目简介

**DeepSea Sediment Classifier (DSSC)** 是一款面向海洋地质研究的桌面软件，能够基于多波束后向散射（MBES Backscatter）和浅地层剖面（Sub-Bottom Profiler）声学数据，利用 **MSC-Transformer** 深度学习模型自动识别海底沉积物类型，并生成空间分类图。

> **参考论文**：Wang et al., *"Research on Seabed Sediment Classification Based on the MSC-Transformer and Sub-Bottom Profiler"*, Journal of Marine Science and Engineering (JMSE), 2023, 11(5), 1074.  
> DOI: [10.3390/jmse11051074](https://doi.org/10.3390/jmse11051074)

### 适用场景

| 场景 | 说明 |
|------|------|
| 🚢 **海洋地质调查** | 批量处理测线数据，快速产出沉积物分类图 |
| 🔬 **科研分析** | 对比不同超参数、不同特征组合的分类效果 |
| 📊 **教学演示** | 可视化展示声学数据与分类结果全过程 |
| 🏭 **工程勘察** | 海底管线、风电基座选址的底质预评估 |

---

## ✨ 核心功能

### 📥 多源数据导入
| 数据类型 | 格式 | 说明 |
|---------|------|------|
| 测线数据 | TXT (CSV) | 经度、纬度、深度、反射强度，支持自定义列映射与分隔符 |
| DEM 底图 | GeoTIFF (.tif) | 多波束后向散射镶嵌图，自动地理参考叠加 |
| 浅剖数据 | SEG-Y (.sgy/.segy) | 浅地层剖面仪原始道集，支持道头解析与坐标提取 |
| 模型权重 | PyTorch (.pt) | 已训练分类模型，导入后可直接推理 |

### 📊 多维度可视化
- **航迹地图**：GPS 航迹多色叠加、DEM 底图透明叠加、鼠标悬停交互
- **声学切面**：测线反射强度剖面（原始 vs 去噪对比）、均值/包络/滤波曲线
- **地震剖面**：SEG-Y 浅剖堆叠显示、专业色标、多文件切换
- **训练监控**：实时 Loss / Accuracy 曲线、Progress Bar + ETA 预估

### 🧠 AI 分类引擎
- **MSC-Transformer**（TabTransformer 架构）：多头自注意力 + 可学习位置编码
- 支持 5 类沉积物自动识别，可通过配置文件扩展至 10+ 类
- 内嵌 K-Means 聚类伪标签生成，无标注数据也可预训练

### ⚙️ 人机交互
- **训练前环境检查**：自动检测 Python 版本、依赖完整性、GPU 状态
- **可视化超参数调节**：学习率、Epochs、Batch Size、Hidden Dim、Dropout、注意力头数、层数
- **类别勾选定制**：可从 20+ 预定义类别中自由选择目标分类数
- **快速加载已有模型**：导入 .pt 权重文件，跳过训练直接推理

### 🖥️ GPU 加速
- **本地 GPU**：自动检测 nvidia-smi，支持指定显卡（CUDA_VISIBLE_DEVICES）
- **远程 GPU**：SSH + SFTP 上传数据 → 远程训练 → 下载模型 → 自动清理（即用即删，零残留）
- 实时流式回传远程训练日志

### 📤 成果输出
| 输出类型 | 格式 | 内容 |
|---------|------|------|
| GPS 分类图 | TXT (CSV) | 经度、纬度、类别ID、类别名称、置信度、各类概率 |
| 分类统计报告 | TXT | 空间范围、各类样本数及比例、置信度统计 |
| 分类地图 | PNG (150 DPI) | 高分辨率等比例尺地理坐标空间分布图 |
| 模型权重 | PyTorch (.pt) | 最佳验证损失模型，含完整超参数元数据 |

### 🛠️ 专业预处理
- TXT 数据：中值滤波、高斯平滑、中值+高斯综合去噪
- SEG 数据：去均值、中值滤波、高斯平滑、AGC 自动增益控制、道间均衡
- 原始 vs 去噪对比可视

---

## 🚀 快速开始

### 方式一：直接使用 exe（推荐普通用户）

> 📦 **适用于 Windows 用户，无需安装任何 Python 环境或其他依赖。**

1. 从 [Releases](../../releases) 页面下载最新版 `DeepSeaSedimentClassifier.exe`

2. 双击 `DeepSeaSedimentClassifier.exe` 启动软件

3. 按照 [用户操作指南](#-用户操作指南) 开始分类

> **注意**：
> - exe 文件约 2-3 GB（包含 PyTorch / PyQt5 等完整运行时），首次启动可能需要 30-60 秒
> - 360 / Windows Defender 可能误报，请添加信任或暂时关闭
> - 若启动失败，请确认系统为 Windows 10 及以上 64 位版本

### 方式二：从源码运行（推荐开发者）

#### 环境要求

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.9+ | 推荐 3.10 |
| pip | 21.0+ | |
| Git | 任意 | 用于克隆仓库 |
| CUDA | 11.8 (可选) | GPU 加速 |

#### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/HanYi-study/DeepSeaSedimentClassifier.git
cd DeepSeaSedimentClassifier

# 2. 创建虚拟环境（推荐）
python -m venv venv

# Windows 激活
venv\Scripts\activate

# Linux/Mac 激活
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. [可选] 如需 GPU 加速，安装 CUDA 版 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. [可选] 安装附加依赖以启用全部功能
pip install rasterio      # DEM GeoTIFF 支持
pip install segyio        # SEG-Y 剖面格式支持
pip install fiona         # Shapefile 导出支持
```

#### 启动软件

```bash
python main.py
```

---

## 📄 数据格式说明

### 测线 TXT（必需）

每行一个采样点，默认列顺序：**经度, 纬度, 深度(m), 反射强度(dB)**

```csv
110.12345,17.56789,150.5,-25.3
110.12346,17.56790,151.2,-26.1
110.12347,17.56791,151.8,-24.8
```

> 💡 可在导入面板中自定义各列的列号（从 1 开始），适配不同格式的数据。

### DEM 底图（可选）

标准 GeoTIFF 格式（`.tif` / `.tiff`），须包含地理参考信息（投影/仿射变换），用于提取高程和坡度特征辅助分类。

### SEG-Y 剖面（可选）

支持标准 SEG-Y 格式（`.sgy` / `.segy`），以及文本矩阵格式。道头需包含坐标信息（Source X/Y），否则将使用默认位置。

---

## 📋 用户操作指南

### 基本工作流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. 导入数据   │ →  │ 2. 提取特征   │ →  │ 3. 配置参数   │
│ TXT / DEM /   │    │ F5 / 菜单     │    │ 学习率/Epoch/ │
│ SEG-Y         │    │               │    │ Batch Size等  │
└──────────────┘    └──────────────┘    └──────────────┘
                                               ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 6. 导出结果   │ ←  │ 5. 查看结果   │ ←  │ 4. 训练/推理  │
│ TXT/PNG/.pt  │    │ 地图+统计     │    │ F6 开始训练   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 快捷操作一览

| 快捷键 | 功能 |
|--------|------|
| `F5` | 提取特征 |
| `F6` | 开始训练 |
| `F7` | 停止训练 |
| `Ctrl+S` | 保存模型 (.pt) |
| `Ctrl+O` | 加载模型 (.pt) |
| `Ctrl+E` | 导出分类结果 |
| `Ctrl+D` | 导入 DEM 底图 |
| `Ctrl+T` | 导入测线 TXT |
| `Ctrl+Q` | 退出 |

### 参数参考

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|---------|------|
| Learning Rate | 0.001 | 1e-4 ~ 1e-2 | AdamW 初始学习率 |
| Epochs | 30 | 10 ~ 200 | 训练轮数，小数据集可增大 |
| Batch Size | 512 | 64 ~ 2048 | 根据内存/显存调整 |
| Hidden Dim | 128 | 64 ~ 512 | 越大模型容量越大 |
| Num Heads | 4 | 2 ~ 8 | 多头自注意力头数 |
| Num Layers | 2 | 1 ~ 6 | Transformer 编码器层数 |
| Dropout | 0.1 | 0.0 ~ 0.5 | 正则化，防过拟合 |
| Train Split | 0.8 | 0.6 ~ 0.9 | 训练/验证集划分比例 |

---

## 🗺️ 沉积物分类体系

| ID | 中文名称 | 英文名称 | 图例 |
|----|---------|---------|------|
| 1 | 钙质生物粉砂 | Calcareous bio-silt | 🟡 |
| 2 | 钙质生物黏土质粉砂 | Calcareous bio-clay silt | 🟢 |
| 3 | 粉砂质砂 | Silty sand | 🔵 |
| 4 | 中砂 | Medium sand | 🟠 |
| 5 | 砾砂 | Gravel sand | 🔴 |

> 💡 内置 20+ 预定义类别，可在 GUI 中自由勾选。修改 `config/settings.py` 中的 `SEDIMENT_CLASSES` 字典可定制。

---

## 🏗️ 模型架构

### MSC-Transformer

```
输入特征 (7维 / 14维含SEG)
       │
       ▼
┌─────────────────────┐
│   Input Embedding    │  Linear(input_dim → hidden_dim)
│   + LayerNorm + GELU │
├─────────────────────┤
│      Dropout         │  p = 0.1
├─────────────────────┤
│  Position Encoding   │  可学习位置编码
├─────────────────────┤
│  Transformer Encoder  │  N 层 × H 头 Multi-Head Self-Attention
│       × N            │  + Feed-Forward (hidden_dim × 4)
├─────────────────────┤
│  Attention Pooling    │  加权池化 (全局特征聚合)
├─────────────────────┤
│     Classifier       │  LayerNorm → Linear → GELU → Dropout
│                      │  → Linear → GELU → Linear(num_classes)
└─────────────────────┘
       │
       ▼
  N 类沉积物概率
```

### 特征工程

| 数据源 | 维度 | 特征内容 |
|--------|------|---------|
| TXT 测线（基础） | 7 | 反射强度、深度、强度梯度、局部粗糙度、平滑强度、经度、纬度 |
| TXT + DEM | 9 | 上述 7 维 + 高程、坡度 |
| TXT + SEG | 14 | 上述 7 维 + 振幅统计(mean/std/max/RMS)、偏度、峰度、总能量、峰值位置、过零率、分层能量比、反射界面梯度 |
| TXT + DEM + SEG | 16 | 全部特征融合 |

---

## 📁 项目结构

```
DeepSeaSedimentClassifier/
│
├── main.py                           # 程序入口（双击或 python main.py）
├── requirements.txt                  # Python 依赖清单
├── build_exe.py                      # PyInstaller 打包脚本
├── DeepSeaSedimentClassifier.spec    # PyInstaller 规格文件
├── runtime_hook.py                   # 打包运行时钩子（DLL 加载顺序修复）
├── icon.ico                          # 软件图标
├── DSSC.vbs                          # Windows 无控制台启动脚本
├── README.md                         # 本文档
│
├── config/                           # 全局配置
│   ├── __init__.py
│   └── settings.py                   # 超参数默认值、沉积物类别、颜色映射等
│
├── core/                             # 核心业务逻辑模块
│   ├── __init__.py
│   ├── data_loader.py                # 多格式数据导入 (TXT / TIF / SEG-Y)
│   ├── data_processor.py             # 特征提取与标准化
│   ├── classifier.py                 # 分类器封装（训练/推理/保存/加载）
│   ├── seg_reader.py                 # SEG-Y 道集读取与道头解析
│   ├── seg_processor.py              # SEG 特征提取
│   ├── seg_preprocess.py             # SEG 去噪预处理（去均值/滤波/AGC/道均衡）
│   ├── backscatter_profile.py        # 后向散射声学切面构建
│   ├── gpu_manager.py                # GPU 管理（本地检测 + 远程 SSH 训练）
│   ├── env_checker.py                # 运行时环境检查
│   ├── exporter.py                   # 成果导出（TXT / CSV / PNG / SHP）
│   └── remote_train_script.py        # 远程 GPU 训练脚本
│
├── models/                           # 深度学习模型定义
│   ├── __init__.py
│   └── msc_transformer.py            # MSC-Transformer 模型（PyTorch）
│
├── ui/                               # PyQt5 图形用户界面
│   ├── __init__.py
│   └── main_window.py                # 主窗口（TabWidget / 工具栏 / 菜单栏）
│
├── utils/                            # 工具模块
│   ├── __init__.py
│   └── logger.py                     # 日志系统
│
├── tests/                            # 测试与示例数据
│   ├── generate_sample_data.py       # 示例数据生成脚本
│   ├── sample_profile.txt            # 示例测线数据
│   └── sample_survey*.txt            # 示例多测线数据（5条）
│
├── docs/                             # 文档
│   └── 软件功能说明与标书正文.md        # 功能详细说明
│
├── dist/                             # 打包输出目录
│   └── DeepSeaSedimentClassifier.exe # 独立可执行文件
│
├── best_model.pt                     # 本地训练的最佳模型
└── best_model_remote.pt              # 远程训练的最佳模型
```

---

## 🔧 开发环境搭建

### 推荐工具链

| 工具 | 用途 |
|------|------|
| VS Code / PyCharm | IDE |
| Python 3.10 | 运行时 |
| Git | 版本控制 |
| CUDA 11.8 Toolkit | GPU 加速（可选） |

### 安装开发依赖

```bash
pip install -r requirements.txt

# 开发辅助工具
pip install pytest black
```

### 运行测试

```bash
# 生成测试数据
python tests/generate_sample_data.py

# 启动软件验证
python main.py
```

---

## 🛠️ 开发指南

本项目采用**模块化架构**，核心逻辑与 UI 分离，便于二次开发与功能扩展。

### 修改默认超参数

编辑 `config/settings.py`，修改对应变量即可，无需改动任何业务代码。

```python
# config/settings.py
DEFAULT_LEARNING_RATE = 0.0005   # 改默认学习率
DEFAULT_HIDDEN_DIM = 256          # 改默认隐藏维度
SEDIMENT_CLASSES = {1: "新类别1", 2: "新类别2"}  # 自定义类别
```

### 添加新的特征

在 `core/data_processor.py` 的 `extract_features()` 方法中添加新的特征列，并更新 `feature_names` 列表：

```python
# core/data_processor.py
def extract_features(self, survey_lines, dem=None):
    # ... 原有代码 ...
    # 添加新特征
    new_feature = calculate_my_feature(intensity)
    features = np.column_stack([
        intensity, depth, gradient, roughness,
        smooth_intensity, lon, lat,
        new_feature,  # ← 新增
    ])
    feature_names = [..., "我的新特征"]
```

### 替换分类模型

1. 在 `models/` 目录创建新模型文件（如 `my_model.py`）
2. 在 `core/classifier.py` 中导入并使用
3. 确保新模型实现 `forward()` / `predict()` / `predict_proba()` 接口

### 添加新的数据格式

在 `core/data_loader.py` 的 `DataLoader` 类中添加 `load_xxx()` 方法：

```python
def load_netcdf(self, file_path: str) -> Optional[SurveyLine]:
    """加载 NetCDF 格式数据"""
    # 实现解析逻辑
    pass
```

### 添加新的导出格式

在 `core/exporter.py` 的 `Exporter` 类中添加 `export_xxx()` 方法：

```python
def export_geojson(self, data, output_path):
    """导出为 GeoJSON 格式"""
    # 实现导出逻辑
    pass
```

### 架构约定

- **数据流方向**：`data_loader` → `data_processor` → `classifier` → `exporter`
- **UI 与核心分离**：UI 层不直接操作模型张量，通过 `SedimentClassifier` API 调用
- **日志规范**：使用 `utils.logger.logger` 统一记录，不直接 `print()`
- **异常处理**：核心模块抛出明确异常，UI 层捕获并提示用户

---

## 📦 打包为独立 exe

```bash
# 确保已安装 PyInstaller
pip install pyinstaller

# 执行打包（约 5-10 分钟）
python build_exe.py

# 输出位置
# dist/DeepSeaSedimentClassifier.exe  (~2-3 GB)
```

> ⚠️ **注意事项**：
> - 打包前请先确认 `build_exe.py` 中的 Python 路径与当前环境一致
> - `--onefile` 模式将全部依赖打包为单文件，方便分发但启动较慢
> - 如需缩小体积，可改用 `--onedir` 模式（目录输出，启动更快）
> - 首次打包建议关闭杀毒软件，避免误删临时文件

---

## ❓ 常见问题

<details>
<summary><strong>Q: 没有 GPU 能跑吗？</strong></summary>

**A:** 完全可以。软件会自动检测并使用 CPU 训练，功能完整无缺，仅训练速度较慢。对于几千到几万条样本的数据集，CPU 训练通常在可接受范围内。
</details>

<details>
<summary><strong>Q: 没有真实标注数据怎么办？</strong></summary>

**A:** 软件内置 K-Means 聚类自动生成伪标签功能。您可以在无标注数据上先做无监督预训练，获得初步分类结果后再进行人工修正。
</details>

<details>
<summary><strong>Q: 如何获得更好的分类效果？</strong></summary>

- 导入 DEM 底图可提取地形特征（高程、坡度），辅助分类
- 导入 SEG-Y 浅剖数据可提取 13 维声学特征，显著提升区分度
- 增大 `hidden_dim`（如 256 → 512）和 `num_layers`（如 2 → 4）
- 减小 `learning_rate` 并增大 `epochs`
- 检查数据质量：去噪前/后对比，异常值剔除
</details>

<details>
<summary><strong>Q: 如何连接远程 GPU 服务器？</strong></summary>

1. 确保远程服务器已安装 Python 3、PyTorch（CUDA版）和 scikit-learn
2. 在软件 "GPU 配置" 面板填写：IP 地址、用户名、密码、端口
3. 点击 "检测远程 GPU" 确认连接成功
4. 点击 "使用远程 GPU 训练"，系统自动完成：上传数据 → 训练 → 下载模型 → 清理临时文件
</details>

<details>
<summary><strong>Q: exe 被杀毒软件拦截怎么办？</strong></summary>

**A:** PyInstaller 打包的单文件 exe 可能被部分杀毒软件误报。请将 exe 所在目录添加到白名单，或在 Windows Defender → "病毒和威胁防护" → "排除项" 中添加。
</details>

<details>
<summary><strong>Q: Linux 上能用吗？</strong></summary>

**A:** exe 仅适用于 Windows。但在 Linux 上可以直接从源码运行（`python main.py`），功能完全一致。在 Linux 上使用 PyInstaller 也可以打包为 Linux 可执行文件。
</details>

---

## 📚 引用

如果您在研究中使用了本软件，请引用以下论文：

```bibtex
@Article{jmse11051074,
  AUTHOR = {Wang, et al.},
  TITLE  = {Research on Seabed Sediment Classification Based on
            the MSC-Transformer and Sub-Bottom Profiler},
  JOURNAL= {Journal of Marine Science and Engineering},
  VOLUME = {11},
  YEAR   = {2023},
  NUMBER = {5},
  DOI    = {10.3390/jmse11051074}
}
```

---

## 📜 许可证

本软件仅供学术研究使用。商业用途请联系作者获取授权。

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- **GitHub Issues**：[提交 Issue](../../issues)
- **Email**：请通过项目主页获取联系方式

---

<p align="center">
  <sub>Made with ❤️ for marine geology research</sub>
</p>
