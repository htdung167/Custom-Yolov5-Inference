python export.py --data ./data/badminton.yaml \
                --weights ../../weights/best.pt \
                --img 416 416 --batch 1 --opset 12 \
                --dynamic --simplify \
                --include "onnx" 
