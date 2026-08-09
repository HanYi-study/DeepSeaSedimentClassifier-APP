"""
Backscatter Profile Generator
==============================
从 TIF 底图或 TXT 测线数据中提取/生成沿航迹反射强度剖面。

输入: TIF (可选) + TXT 测线 (GPS + 强度 + 序号)
输出: BackscatterProfile (距离 + 强度 + 可用于分类的特征)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BackscatterProfile:
    """沿航迹反射强度剖面"""
    distance: np.ndarray          # 沿航迹累积距离 (m)
    intensity: np.ndarray         # 反射强度
    lon: np.ndarray               # 经度
    lat: np.ndarray               # 纬度
    seq: np.ndarray               # 采样序号
    profile_2d: np.ndarray        # 2D 剖面图像 (用于显示)
    extent: tuple                 # imshow extent


def _haversine_distance(lon, lat):
    """Haversine 公式计算沿航迹累积距离 (m)"""
    R = 6371000.0
    rlon = np.radians(np.asarray(lon, float))
    rlat = np.radians(np.asarray(lat, float))
    dlon = np.diff(rlon); dlat = np.diff(rlat)
    a = np.sin(dlat/2)**2 + np.cos(rlat[:-1])*np.cos(rlat[1:])*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a+1e-15))
    d = np.zeros(len(lon))
    d[1:] = np.cumsum(R*c)
    return d


class TifBackscatterReader:
    """从 GeoTIFF 底图中按 GPS 坐标提取反射强度"""

    def __init__(self, tif_path):
        import rasterio
        try:
            from pyproj import Transformer
        except ImportError:
            raise ImportError(
                "TIF reading requires pyproj. Install: pip install pyproj\n"
                "Or use TXT-only mode (TIF is optional)."
            )

        self.ds = rasterio.open(tif_path)
        self.data = self.ds.read(1).astype(np.float32)
        self.transform = self.ds.transform
        self.crs = self.ds.crs
        # WGS84 → TIF CRS
        self.transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def sample(self, lon, lat):
        """从 TIF 中提取 (lon, lat) 处的反射强度"""
        from rasterio.transform import rowcol
        x, y = self.transformer.transform(lon, lat)
        r, c = rowcol(self.transform, x, y)
        if 0 <= r < self.data.shape[0] and 0 <= c < self.data.shape[1]:
            return self.data[r, c]
        return np.nan

    def sample_track(self, lon_arr, lat_arr):
        """沿航迹批量提取"""
        return np.array([self.sample(lon_arr[i], lat_arr[i])
                         for i in range(len(lon_arr))], dtype=np.float32)

    def close(self):
        self.ds.close()


def generate_profile_from_txt(
    lon, lat, intensity, seq,
    profile_height: int = 80,
) -> BackscatterProfile:
    """
    从 TXT 测线数据生成反射强度剖面。

    Parameters
    ----------
    lon, lat : 经纬度
    intensity : 反射强度值
    seq : 采样序号 (深度代理)
    profile_height : 2D 图像高度 (像素)

    Returns
    -------
    BackscatterProfile
    """
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    intensity = np.asarray(intensity, np.float64)
    seq = np.asarray(seq, np.float64)

    distance = _haversine_distance(lon, lat)

    # 归一化强度到 [0, 1]
    i_min, i_max = np.nanpercentile(intensity, [1, 99])
    if i_max - i_min < 1e-8:
        i_min, i_max = intensity.min(), intensity.max() + 1e-8
    intensity_norm = np.clip((intensity - i_min) / (i_max - i_min), 0, 1)

    # 生成 2D 剖面图像 (深度方向复制, 模拟地层条带)
    profile_2d = np.tile(intensity_norm, (profile_height, 1))

    return BackscatterProfile(
        distance=distance,
        intensity=intensity,
        lon=lon,
        lat=lat,
        seq=seq,
        profile_2d=profile_2d,
        extent=(distance[0], distance[-1], profile_height, 0),
    )


def generate_profile_from_tif(
    tif_path, lon, lat,
    profile_height: int = 80,
) -> BackscatterProfile:
    """
    从 TIF 底图 + GPS 轨迹提取反射强度剖面。
    """
    reader = TifBackscatterReader(tif_path)
    intensity = reader.sample_track(lon, lat)
    reader.close()

    # 填充 NaN
    mask = np.isnan(intensity)
    if mask.any():
        intensity[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask),
            intensity[~mask],
        ) if (~mask).sum() > 1 else 0

    return generate_profile_from_txt(lon, lat, intensity,
                                     np.arange(len(lon)), profile_height)
