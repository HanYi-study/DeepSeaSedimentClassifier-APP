"""
数据预处理模块
============
对导入的测线数据进行清洗、特征提取、标准化等预处理操作。
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from core.data_loader import SurveyLine, DEMData
from utils.logger import logger


@dataclass
class ProcessedData:
    """预处理后的数据"""
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    coordinates: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: List[str] = field(default_factory=list)
    num_samples: int = 0
    num_features: int = 0


class DataProcessor:
    """
    数据预处理器
    ===========
    提供数据清洗、特征提取、标准化等功能。
    可在此类中扩展新的特征提取方法。
    """

    def __init__(self):
        self.processed: Optional[ProcessedData] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

    # ==================== 特征提取 ====================

    def extract_features(
        self,
        survey_lines: List[SurveyLine],
        dem: Optional[DEMData] = None,
    ) -> ProcessedData:
        """
        从测线数据中提取分类特征。

        基本特征：
          - 反射强度 (dB)
          - 深度 (m)
          - 局部梯度 (强度变化率)
          - 局部粗糙度 (滑动窗口标准差)
          - 经度、纬度 (空间位置)

        用户可在此方法中扩展更多特征。

        Parameters
        ----------
        survey_lines : list of SurveyLine
            测线数据列表
        dem : DEMData, optional
            DEM 底图数据

        Returns
        -------
        ProcessedData
        """
        if not survey_lines:
            logger.error("没有可用的测线数据")
            return ProcessedData()

        all_features = []
        all_coords = []

        for sl in survey_lines:
            n = len(sl.longitude)
            if n < 2:
                continue

            # --- 基础特征 ---
            intensity = sl.reflection_intensity.astype(np.float64)
            depth = sl.sequence_number.astype(np.float64)  # 序号=深度代理
            lon = sl.longitude.astype(np.float64)
            lat = sl.latitude.astype(np.float64)

            # --- 派生特征 ---
            # 局部梯度 (相邻点强度差)
            gradient = np.zeros(n)
            gradient[1:] = np.abs(intensity[1:] - intensity[:-1])
            gradient[0] = gradient[1] if n > 1 else 0

            # 局部粗糙度 (5点滑动窗口标准差)
            window = 5
            roughness = np.zeros(n)
            for i in range(n):
                start = max(0, i - window // 2)
                end = min(n, i + window // 2 + 1)
                roughness[i] = np.std(intensity[start:end])

            # 滑动均值 (7点窗口)
            window2 = 7
            smooth_intensity = np.zeros(n)
            for i in range(n):
                start = max(0, i - window2 // 2)
                end = min(n, i + window2 // 2 + 1)
                smooth_intensity[i] = np.mean(intensity[start:end])

            # --- 组装特征矩阵 ---
            features = np.column_stack([
                intensity,        # 反射强度
                depth,            # 深度
                gradient,         # 强度梯度
                roughness,        # 局部粗糙度
                smooth_intensity, # 平滑强度
                lon,              # 经度
                lat,              # 纬度
            ])
            coords = np.column_stack([lon, lat])

            all_features.append(features)
            all_coords.append(coords)

        if not all_features:
            logger.error("特征提取后无有效数据")
            return ProcessedData()

        combined_features = np.vstack(all_features)
        combined_coords = np.vstack(all_coords)

        feature_names = [
            "反射强度 (dB)",
            "深度 (m)",
            "强度梯度",
            "局部粗糙度",
            "平滑强度",
            "经度",
            "纬度",
        ]

        # 移除 NaN
        valid_mask = ~np.isnan(combined_features).any(axis=1)
        if not valid_mask.all():
            n_removed = (~valid_mask).sum()
            logger.warning(f"移除 {n_removed} 个含 NaN 的样本")
            combined_features = combined_features[valid_mask]
            combined_coords = combined_coords[valid_mask]

        self.processed = ProcessedData(
            features=combined_features,
            coordinates=combined_coords,
            feature_names=feature_names,
            num_samples=combined_features.shape[0],
            num_features=combined_features.shape[1],
        )

        logger.info(
            f"特征提取完成: {self.processed.num_samples} 条记录, "
            f"{self.processed.num_features} 个特征"
        )
        logger.info(f"特征列表: {', '.join(feature_names)}")
        return self.processed

    # ==================== 标准化 ====================

    def normalize(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Z-score 标准化。

        Parameters
        ----------
        features : np.ndarray
            待标准化特征
        fit : bool
            True=计算均值/标准差, False=使用已有参数

        Returns
        -------
        np.ndarray
        """
        if fit:
            self._feature_mean = features.mean(axis=0)
            self._feature_std = features.std(axis=0)
            self._feature_std[self._feature_std == 0] = 1.0  # 避免除零

        if self._feature_mean is None or self._feature_std is None:
            logger.warning("尚未计算标准化参数，跳过标准化")
            return features

        return (features - self._feature_mean) / self._feature_std

    def get_normalized_features(self) -> np.ndarray:
        """获取标准化后的特征"""
        if self.processed is None:
            return np.array([])
        return self.normalize(self.processed.features.copy(), fit=True)

    # ==================== 数据增强 ====================

    def add_noise(self, features: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        添加高斯噪声进行数据增强。

        Parameters
        ----------
        features : np.ndarray
            输入特征
        noise_level : float
            噪声标准差相对值

        Returns
        -------
        np.ndarray
        """
        noise = np.random.randn(*features.shape) * noise_level * np.std(features, axis=0)
        return features + noise

    # ==================== DEM 特征插值 ====================

    def extract_dem_features(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        dem: DEMData,
    ) -> np.ndarray:
        """
        从 DEM 中提取对应坐标处的高程/坡度特征。

        Parameters
        ----------
        lons, lats : np.ndarray
            坐标数组
        dem : DEMData
            DEM 数据

        Returns
        -------
        np.ndarray (N, 2) -> [elevation, slope]
        """
        if dem is None or dem.raster.size == 0:
            return np.zeros((len(lons), 2))

        try:
            import rasterio
            from rasterio.transform import rowcol

            elevations = np.zeros(len(lons))
            for i, (lon, lat) in enumerate(zip(lons, lats)):
                try:
                    r, c = rowcol(dem.transform, lon, lat)
                    if 0 <= r < dem.raster.shape[0] and 0 <= c < dem.raster.shape[1]:
                        elevations[i] = dem.raster[r, c]
                except Exception:
                    elevations[i] = 0

            # 计算坡度
            dy, dx = np.gradient(dem.raster)
            slope = np.sqrt(dx**2 + dy**2)
            slopes = np.zeros(len(lons))
            for i, (lon, lat) in enumerate(zip(lons, lats)):
                try:
                    r, c = rowcol(dem.transform, lon, lat)
                    if 0 <= r < slope.shape[0] and 0 <= c < slope.shape[1]:
                        slopes[i] = slope[r, c]
                except Exception:
                    slopes[i] = 0

            return np.column_stack([elevations, slopes])

        except ImportError:
            logger.warning("rasterio 未安装，无法提取 DEM 特征")
            return np.zeros((len(lons), 2))
