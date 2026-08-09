"""
成果输出模块
==========
将分类结果导出为 GPS 标记的 TXT 分类图。
支持多种导出格式。
"""

import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from config.settings import (
    SEDIMENT_CLASSES,
    EXPORT_DELIMITER,
    EXPORT_ENCODING,
)
from utils.logger import logger


class Exporter:
    """
    分类结果导出器
    =============
    将分类结果输出为不同格式的文件。

    使用示例:
      exporter = Exporter()
      exporter.export_classification_txt(coordinates, predictions, "output.txt")
    """

    def __init__(self):
        pass

    # ==================== 主要导出: GPS 分类 TXT ====================

    def export_classification_txt(
        self,
        coordinates: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        output_path: str = None,
        include_header: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """
        导出带 GPS 的分类结果 TXT 文件。

        输出格式:
          Longitude, Latitude, Class_ID, Class_Name, Confidence, ...

        Parameters
        ----------
        coordinates : np.ndarray (N, 2)
            经纬度坐标
        predictions : np.ndarray (N,)
            分类标签 (1-based)
        probabilities : np.ndarray (N, C), optional
            各类别概率
        output_path : str, optional
            输出文件路径
        include_header : bool
            是否包含列标题
        include_metadata : bool
            是否包含元数据注释

        Returns
        -------
        str: 输出文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"seafloor_classification_{timestamp}.txt"

        n = len(predictions)

        # 计算置信度 (最大概率)
        if probabilities is not None:
            confidence = np.max(probabilities, axis=1)
        else:
            confidence = np.ones(n)

        # 构建输出数据
        lines = []

        # 元数据注释行
        if include_metadata:
            lines.append(f"# 深海底质分类结果")
            lines.append(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"# 样本数: {n}")
            lines.append(f"# 类别体系:")
            for cls_id, cls_name in SEDIMENT_CLASSES.items():
                count = np.sum(predictions == cls_id)
                lines.append(f"#   {cls_id}: {cls_name} (n={count})")
            lines.append("#")

        # 列标题
        if include_header:
            header_cols = ["Longitude", "Latitude", "Class_ID", "Class_Name", "Confidence"]
            if probabilities is not None:
                for cls_id in range(1, probabilities.shape[1] + 1):
                    header_cols.append(f"Prob_Class_{cls_id}")
            lines.append(EXPORT_DELIMITER.join(header_cols))

        # 数据行
        for i in range(n):
            lon, lat = coordinates[i]
            cls_id = int(predictions[i])
            cls_name = SEDIMENT_CLASSES.get(cls_id, f"Unknown_{cls_id}")
            conf = confidence[i]

            row = [f"{lon:.6f}", f"{lat:.6f}", str(cls_id), cls_name, f"{conf:.4f}"]

            if probabilities is not None:
                for p in probabilities[i]:
                    row.append(f"{p:.4f}")

            lines.append(EXPORT_DELIMITER.join(row))

        # 写入文件
        with open(output_path, "w", encoding=EXPORT_ENCODING) as f:
            f.write("\n".join(lines))

        logger.info(f"分类结果已导出: {output_path} ({n} 条记录)")
        return output_path

    # ==================== 统计报告导出 ====================

    def export_statistics_report(
        self,
        coordinates: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        output_path: str = None,
    ) -> str:
        """
        导出分类统计报告。

        Returns
        -------
        str: 报告文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"classification_report_{timestamp}.txt"

        n = len(predictions)
        lines = []
        lines.append("=" * 60)
        lines.append("           深海底质分类 - 统计报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"总样本数: {n}")
        lines.append("")

        # 空间范围
        lines.append(f"经度范围: [{coordinates[:, 0].min():.6f}, {coordinates[:, 0].max():.6f}]")
        lines.append(f"纬度范围: [{coordinates[:, 1].min():.6f}, {coordinates[:, 1].max():.6f}]")
        lines.append("")

        # 分类统计
        lines.append("-" * 40)
        lines.append(f"{'类别':<30} {'数量':>8} {'比例':>10}")
        lines.append("-" * 40)
        for cls_id, cls_name in SEDIMENT_CLASSES.items():
            count = np.sum(predictions == cls_id)
            ratio = count / n * 100
            lines.append(f"{cls_name:<30} {count:>8} {ratio:>9.1f}%")
        lines.append("-" * 40)

        # 置信度统计
        if probabilities is not None:
            lines.append("")
            lines.append("置信度统计:")
            conf = np.max(probabilities, axis=1)
            lines.append(f"  平均置信度: {conf.mean():.4f}")
            lines.append(f"  最低置信度: {conf.min():.4f}")
            lines.append(f"  最高置信度: {conf.max():.4f}")
            lines.append(f"  低置信度样本(<0.5): {(conf < 0.5).sum()} ({(conf < 0.5).sum()/n*100:.1f}%)")

        lines.append("")
        lines.append("=" * 60)

        with open(output_path, "w", encoding=EXPORT_ENCODING) as f:
            f.write("\n".join(lines))

        logger.info(f"统计报告已导出: {output_path}")
        return output_path

    # ==================== Shapefile 导出 (可选) ====================

    def export_shapefile(
        self,
        coordinates: np.ndarray,
        predictions: np.ndarray,
        output_path: str = None,
    ) -> Optional[str]:
        """
        导出为 ESRI Shapefile 格式（需要 fiona 库）。

        Parameters
        ----------
        coordinates : np.ndarray (N, 2)
        predictions : np.ndarray (N,)
        output_path : str, optional

        Returns
        -------
        str or None
        """
        try:
            import fiona
            from fiona.crs import from_epsg

            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"seafloor_classification_{timestamp}.shp"

            schema = {
                "geometry": "Point",
                "properties": {
                    "Class_ID": "int",
                    "Class_Name": "str",
                },
            }

            with fiona.open(
                output_path,
                "w",
                driver="ESRI Shapefile",
                schema=schema,
                crs=from_epsg(4326),
                encoding=EXPORT_ENCODING,
            ) as dst:
                for i in range(len(predictions)):
                    cls_id = int(predictions[i])
                    dst.write(
                        geometry={"type": "Point", "coordinates": (float(coordinates[i, 0]), float(coordinates[i, 1]))},
                        properties={
                            "Class_ID": cls_id,
                            "Class_Name": SEDIMENT_CLASSES.get(cls_id, f"Unknown"),
                        },
                    )

            logger.info(f"Shapefile 已导出: {output_path}")
            return output_path

        except ImportError:
            logger.warning("fiona 未安装，无法导出 Shapefile。安装: pip install fiona")
            return None
        except Exception as e:
            logger.exception(f"Shapefile 导出失败: {e}")
            return None

    # ==================== CSV 导出 ====================

    def export_csv(
        self,
        coordinates: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        output_path: str = None,
    ) -> str:
        """
        导出为标准 CSV 格式。
        与 export_classification_txt 相同但强制 CSV 扩展名。
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"seafloor_classification_{timestamp}.csv"

        return self.export_classification_txt(
            coordinates=coordinates,
            predictions=predictions,
            probabilities=probabilities,
            output_path=output_path,
            include_header=True,
            include_metadata=False,
        )
