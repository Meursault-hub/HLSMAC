# StarCraft II 多智能体强化学习 (SMAC) 运行说明

本文档详细说明了如何配置环境以及针对不同地图（Map）启动训练或测试命令。

## 1. 环境预配置 (关键步骤)

在运行任何命令之前，**必须**执行以下文件覆盖操作，以确保环境文件是最新的或适配特定战术地图的：

请将项目根目录下 `sc2_tactics` 文件夹中的所有文件，复制并覆盖到 `smac` 库的对应目录中：

*   **源目录:** `./sc2_tactics/`
*   **目标目录:** `.../site-packages/smac/env/sc2_tactics/`

> **注意**：请确保您已安装 `smac` 环境，并且找对了库的安装路径。


## 2. best_model运行命令

根据地图类型的不同，需要使用不同的算法配置文件 (`config`)。

适用于以下地图：
*   **gmzz** (关门打狗)
*   **wzsy** (围魏救赵)
*   **dhls** (调虎离山)
*   **sdjx** (声东击西)

**启动命令格式：**

```bash
python src/main.py --config=qatten --env-config=sc2te with env_args.map_name=[地图代号]_te checkpoint_path="results/models/best_model_[地图代号]"
```

适用于除上述四张地图以外的其他所有地图。

**启动命令格式：**

```bash
python src/main.py --config=qatten_new --env-config=sc2te with env_args.map_name=[地图代号]_te checkpoint_path="results/models/best_model_[地图代号]"
```


## 3. 日志与结果

*   **日志文件**： 所有的运行日志和实验数据将保存在 `result/sacred` 目录下。
*   **模型路径**： 命令中指定的 `checkpoint_path` 用于加载或保存最佳模型。