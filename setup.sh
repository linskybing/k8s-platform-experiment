#! /bin/bash

sudo apt update
sudo apt install -y libgl1

yolo train model=yolo11n.pt data=coco8.yaml epochs=3 imgsz=640