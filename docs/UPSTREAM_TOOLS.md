# 上游工具与第三方代码

本仓库的自研代码可以与下列钙成像工具配合使用，但不直接维护、打包或再分发它们的源码。请从上游项目安装与引用，并以相应版本的文档和许可证为准。

| 工具 | 上游仓库 | 在研究流程中的角色 |
| --- | --- | --- |
| CaImAn | [flatironinstitute/CaImAn](https://github.com/flatironinstitute/CaImAn) | 运动校正、源分离、去卷积 |
| suite2p | [MouseLand/suite2p](https://github.com/MouseLand/suite2p) | 配准、ROI 检测、活动轨迹提取 |
| DeepCAD | [cabooster/DeepCAD](https://github.com/cabooster/DeepCAD) | 自监督钙成像去噪 |
| DeepCAD-RT | [cabooster/DeepCAD-RT](https://github.com/cabooster/DeepCAD-RT) | DeepCAD 的实时/低内存实现 |
| DeepInterpolation | [AllenInstitute/deepinterpolation](https://github.com/AllenInstitute/deepinterpolation) | 时序插值去噪 |

历史版本曾把这些项目的完整源码副本放在仓库根目录。公开版通过 `.gitignore` 排除这些本地检出目录，以减少仓库体积并避免把不同项目的许可范围混在一起。与两套主流 ROI/源分离工具的能力边界研究记录见 [钙波提取算法研究](research-notes/calcium-wave-extraction.md)。

## 引用与复现

- 论文或报告应同时引用实际使用的上游软件、版本、参数与对应方法论文。
- 不要仅写“使用本平台完成分析”；应记录数据预处理、事件阈值、随机种子和模型配置。
- 本仓库中的方法组合不能替代对输出的神经科学解释、统计验证和人工质量控制。
