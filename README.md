## 运行
- 执行如下命令安装transformers等相关依赖
```bash
pip3 install -e ./transformers
```

- 下载模型

- 执行python test.py

## 导出ONNX
- 执行如下命令导出ONNX，可根据实际情况修改导出参数
```bash
python3 export_qwen2vl_onnx.py
```

- 可执行如下命令使用ONNX进行推理，可根据实际情况修改运行参数
```bash
python3 qwen2_vl_onnx_infer.py
```