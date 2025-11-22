import torch
from torchvision import models

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available.")

def list_torchvision_models():
    print("Available models in torchvision:")
    for model_name in dir(models):
        if callable(getattr(models, model_name)) and not model_name.startswith("_"):
            print(model_name)

def list_detection_segmentation_keypoint_models():
    print("\nObject Detection Models:")
    detection_models = [
        "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_mobilenet_v3_large_fpn",
        "retinanet_resnet50_fpn"
    ]
    print("\n".join(detection_models))

    print("\nSemantic Segmentation Models:")
    segmentation_models = [
        "deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"
    ]
    print("\n".join(segmentation_models))

    print("\nKeypoint Detection Models:")
    keypoint_models = ["keypointrcnn_resnet50_fpn"]
    print("\n".join(keypoint_models))

if __name__ == "__main__":
    check_gpu()
    list_torchvision_models()
    print("\n Other models:\n")
    list_detection_segmentation_keypoint_models()