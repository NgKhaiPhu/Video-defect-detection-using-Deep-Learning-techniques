from FlickerDetection.models.resnet_nonlocal import get_model
from FlickerDetection.inference import inference

model = get_model('./checkpoints/r3d101_K_200ep.pth',pretrained=True)
inference('./dataset/data/CP map CP.mp4', './test.avi', model)