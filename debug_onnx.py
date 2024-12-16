import onnx
ret = onnx.checker.check_model("models/onnx/vit/vision_transformer.onnx")
model = onnx.load('models/onnx/vit/vision_transformer.onnx')
print(model.graph.input)
print(model.graph.output)

si = 0
for idx, n in enumerate(model.graph.node):
    if n.name == "/Slice_12":
        print(idx, n)
        si = idx
        break
for i in range(si-50, si):
    print(model.graph.node[i])