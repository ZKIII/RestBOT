import torch
import random
import numpy as np
from ultralytics import YOLO


class RestBotModel:
    def __init__(self, model_name, abs_project_path):
        self.model = YOLO(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.abs_project_path = abs_project_path

    def train_model(self, epochs, batch, imgsz, optimizer, lr):
        self.model.train(data=f"{self.abs_project_path}/datasets/final_dataset/data.yaml",
                         epochs=epochs,
                         batch=batch,
                         imgsz=imgsz,
                         optimizer=optimizer,
                         lr0=lr,
                         device=self.device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(1)

    epochs = 100
    batch = 4
    imgsz = 640
    optimizer = 'Adam'
    lr = 0.001

    rest_bot_model = RestBotModel(model_name="yolov8n.pt",
                                  abs_project_path="-----replace-----")
    rest_bot_model.train_model(epochs, batch, imgsz, optimizer, lr)