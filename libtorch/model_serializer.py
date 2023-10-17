import torch
from torchvision.models import resnet50, densenet121, mobilenet_v2, convnext_small
from torchvision.datasets import CIFAR10
import torchvision

num_classes = len(CIFAR10(root='../datasets/cifar-10-py/', train=True, download=True).classes)

resnet50_mat = resnet50()
resnet50_mat.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)

densenet121_mat = densenet121()
densenet121_mat.classifier = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)

mobilenet_v2_mat = mobilenet_v2()
mobilenet_v2_mat.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)

convnext_small_mat = convnext_small()
convnext_small_mat.classifier[2] = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)

for model, name in [(resnet50_mat, "resnet50"), (densenet121_mat, "densenet121"),
					(mobilenet_v2_mat, "mobilenet_v2"), (convnext_small_mat, "convnext_small_mat")]:
	serialized = torch.jit.script(model)
	serialized.save(f"serialized_models/{name}_for_cifar{num_classes}.pt")