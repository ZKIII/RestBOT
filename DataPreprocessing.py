import os
import json
import yaml
import shutil
from roboflow import Roboflow


class Downloader:

    def __init__(self, api_key: str):
        self.rf = Roboflow(api_key=api_key)
        self.data_params = None

    def download_data(self):
        with open("datasets/data_params.json", "r") as f:
            self.data_params = json.load(f)
            for k in self.data_params.keys():
                project = self.rf.workspace(self.data_params[k]["workspace"]).project(self.data_params[k]["project"])
                version = project.version(self.data_params[k]["version"])
                version.download(self.data_params[k]["model"], location=f"datasets/{k}")


class DataFilter:

    def __init__(self, data_params: dict):
        self.data_params = data_params
        self.yaml_params = None
        self.final_train_images = "datasets/final_dataset/train/images"
        self.final_train_labels = "datasets/final_dataset/train/labels"
        self.final_val_images = "datasets/final_dataset/val/images"
        self.final_val_labels = "datasets/final_dataset/val/labels"

    def create_necessary_files(self):
        os.makedirs(self.final_train_images, exist_ok=True)
        os.makedirs(self.final_train_labels, exist_ok=True)
        os.makedirs(self.final_val_images, exist_ok=True)
        os.makedirs(self.final_val_labels, exist_ok=True)

        classes = []
        for data_path in self.data_params.keys():
            types = set(self.data_params[data_path]["types"].values())
            for c in types:
                match c:
                    case 0:
                        classes.append("Chair")
                    case 1:
                        classes.append("Table")
                    case 2:
                        classes.append("customer")
                    case 3:
                        classes.append("staff")

        self.yaml_params = {
            "nc": len(classes),
            "names": classes,
            "train": "train/images",
            "val": "val/images"
        }

        yaml_path = "datasets/final_dataset/data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(self.yaml_params, f)

    def transfer_single_file(self, data_path, dataset_type, dataset_name):
        img_path = f"{data_path}/{dataset_type}/images"
        imgs = os.listdir(img_path)

        label_path = f"{data_path}/{dataset_type}/labels"
        labels = os.listdir(label_path)

        for i in range(len(imgs)):
            single_img = f"{img_path}/{imgs[i]}"
            if dataset_type == "train":
                shutil.copy2(single_img, self.final_train_images)
                with open(f"{label_path}/{labels[i]}", "r") as read_file, \
                        open(f"{self.final_train_labels}/{labels[i]}", "w") as write_file:
                    for line in read_file:
                        if line[0] in self.data_params[dataset_name]["types"].keys():
                            label = self.data_params[dataset_name]["types"][line[0]]
                            new_line = f"{label}{line[1:]}"
                            write_file.write(new_line)

            elif dataset_type == "test" or dataset_type == "valid":
                shutil.copy2(single_img, self.final_val_images)
                with open(f"{label_path}/{labels[i]}", "r") as read_file, \
                        open(f"{self.final_val_labels}/{labels[i]}", "w") as write_file:
                    for line in read_file:
                        if line[0] in self.data_params[dataset_name]["types"].keys():
                            label = self.data_params[dataset_name]["types"][line[0]]
                            new_line = f"{label}{line[1:]}"
                            write_file.write(new_line)

    def filter_data(self):
        for k in self.data_params.keys():
            data_path = f"datasets/{k}"
            dir_lst = [item for item in os.listdir(data_path) if os.path.isdir(f"{data_path}/{item}")]
            for d in dir_lst:
                self.transfer_single_file(data_path, d, k)


if __name__ == "__main__":
    downloader = Downloader(api_key="-----replace-----")
    downloader.download_data()
    data_filter = DataFilter(data_params=downloader.data_params)
    data_filter.create_necessary_files()
    data_filter.filter_data()