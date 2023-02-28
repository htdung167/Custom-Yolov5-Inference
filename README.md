# Custom-Yolov5-Inference

# How to run
- Git clone: 
```
  git clone https://github.com/htdung167/Custom-Yolov5-Inference.git
```
- cd:
```
  cd Custom-Yolov5-Inference
```

- Download weights and put it in weights folder.

- Create conda env:
```
  conda create -n <env_name> 
  conda activate <env_name>
  conda install pip
  pip install -r requirements.txt
```
- Edit yolov5.cfg file
- Run
```
  bash run_detect.sh
```

- Results will save in experiments
