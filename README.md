# Fictional Map Generator (本地桌面版)

一个面向架空世界创作的本地地图工具，支持从手绘底稿到海岸线、海底地形、群岛与地形细节编辑的完整流程。

## 功能概览

- 底稿绘制：陆地/海洋笔刷、橡皮、封闭区域右键自动填充
- 海岸线生成：基于底稿 + 噪声细化
- 海底地形矩阵：生成 bathymetry（浮点矩阵）并可视化
- 群岛生成：浅海约束 + 岛链分布
- 地形编辑：隆起 / 压低 / 平滑（高度图）
- 视图能力：缩放、平移（手掌工具）、视图切换
- 资源管理：导出资源、保存工程、加载工程

## 技术栈

- Python 3.10+
- PySide6（桌面 GUI）
- NumPy（矩阵与算法）
- Pillow（图像处理与导出）

## 安装与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/app.py
```

## 推荐工作流

1. 在 `底稿` 菜单中先完成大陆轮廓（可用右键自动填充）
2. 在 `生成` 菜单执行：
   - 生成海岸线
   - 生成海底矩阵
   - 生成群岛（可选）
3. 进入 `地形` 菜单：
   - 进入地形编辑
   - 选择隆起/压低/平滑笔刷进行细节雕刻
4. 在 `文件` 菜单导出资源或保存工程

## 快捷操作（常用）

- 撤销：`Cmd/Ctrl + Z`
- 画布平移：空格 + 左键拖动，或中键拖动
- 双击空格：复位缩放与平移
- 直线绘制：按住 `Shift` 再左键绘制

> 更多快捷键可在顶部菜单中查看（菜单项右侧显示）。

## 导出内容

导出目录中可能包含：

- `final_mask.png`：最终陆海二值图
- `bathymetry.npy`：海底地形矩阵
- `terrain_height.npy`：地形高度矩阵（若已编辑）
- `config.json`：本次导出配置

## 工程保存内容

保存工程目录中可能包含：

- `draft_mask.npy`
- `refined_mask.npy`
- `final_mask.npy`
- `bathymetry.npy`
- `terrain_height.npy`
- `project.json`

## 目录结构

```text
src/
  app.py
  ui/
    main_window.py
    canvas_widget.py
  core/
    coastline.py
    bathymetry.py
    islands.py
    terrain.py
    project_io.py
    models.py
    scale.py
  generate_demo.py
  batch_generate.py
```

## 当前状态

当前版本为 MVP，重点在“底稿 -> 生成 -> 细节编辑 -> 导出”闭环。  
后续可继续扩展侵蚀、河流、风格化渲染、神经网络细化等模块。
