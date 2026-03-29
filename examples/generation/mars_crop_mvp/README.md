# SenseNova-MARS-8B crop-only MVP demo
这个版本按仓库里的 tool-mode 约定实现，但只保留 `image_zoom_in_tool`，不依赖 Docker。
## 目录
- `demo_crop_only.py`：最小可跑 demo，仿照 pasted code 做了单次业务调用 + 内部多轮循环
- `tools_crop_only.yaml`：仅保留 crop 工具，工具 schema 与仓库里的 `inference/tools_eval.yaml` 保持一致
- `serve_mars_8b.sh`：本地启动 `sensenova/SenseNova-MARS-8B` 的脚本，端口默认 `8888`
## 安装
```bash
pip install -U openai pillow pyyaml sglang
```
## 启动模型
```bash
bash serve_mars_8b.sh
```
## 运行 demo
```bash
python demo_crop_only.py \
  --question 'What is written below "OLU56130"?' \
  --image /path/to/image.jpg
```
如果你的服务地址不是默认值：
```bash
cd /path/to/mars_crop_mvp

python demo_crop_only.py \
  --question 'What is written below "OLU56130"?' \
  --image /path/to/image.jpg \
  --base-url http://127.0.0.1:8000/v1 \
  --model /data4/home/models/SenseNova-MARS-8B
```
## 行为说明
- system prompt 与仓库测试配置中的 tool-mode prompt 保持一致
- tools section 的拼接方式与仓库 `inference/eval.py` 保持一致
- crop 坐标使用仓库 inference 里的 `0-1000` 相对坐标约定
- crop 后图像处理沿用仓库 inference 的 `smart_resize/process_image` 逻辑
- 其他 search 工具不会暴露给模型；若模型异常输出其他工具调用，会返回 disabled 错误
