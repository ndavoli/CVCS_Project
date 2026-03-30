# CVCS_Project
Computer Vision and Cognitive Systems Exam Project

A comparative study of AlexNet, VGG-16, ResNet-50, and DINOv2 (ViT-B/14) on the PlantVillage dataset (54,305 images, 38 classes). The three CNNs are trained from scratch under uniform conditions, while DINOv2 is evaluated via linear probing on frozen self-supervised features. VGG-16 achieves the highest top-1 accuracy (99.38%), ResNet-50 offers the best efficiency (98.86% with 23.6M parameters), and DINOv2 reaches 98.37% using only a linear head on general-purpose pretrained features.
