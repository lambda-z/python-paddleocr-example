
# 参考资料

###### source: https://aistudio.baidu.com/projectdetail/507159

###### Github: https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph

###### deployment docs: https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/quickstart.md


# 1. 安装PaddlePaddle


```bash
  pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  pip install "paddleocr>=2.0.1"
```

# 2. 初步使用案例

初次使用时会安装依赖包，安装完成后会自动下载预训练模型，模型默认存放在用户目录下的 .paddleocr/whl/ 目录中，可通过修改配置文件中的参数 det_model_dir, rec_model_dir, cls_model_dir 修改模型存放路径。
```bash
paddleocr --image_dir ./imgs/tz.png --use_angle_cls true --use_gpu false
```

结果:
```bash
[2024/01/09 23:04:59] ppocr INFO: **********./imgs/tz.png**********
[2024/01/09 23:05:00] ppocr DEBUG: dt_boxes num : 1, elapsed : 0.6851787567138672
[2024/01/09 23:05:00] ppocr DEBUG: cls num  : 1, elapsed : 0.02373957633972168
[2024/01/09 23:05:00] ppocr DEBUG: rec_res num  : 1, elapsed : 0.15536737442016602
[2024/01/09 23:05:00] ppocr INFO: [[[325.0, 252.0], [945.0, 252.0], [945.0, 572.0], [325.0, 572.0]], ('特征', 0.9942247867584229)]
```
