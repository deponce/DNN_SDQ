import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio


def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')
    image_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info

def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index, layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x

def compute_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])

        result_list = []
        for k in topk:
            # print(corrects[:k].flatten().sum(dtype=torch.float32), batch_size)
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            # result_list.append(correct_k * (100.0 / batch_size))
            result_list.append(correct_k)
        return result_list

def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = int(np.ceil(np.sqrt(feature_map_num)))
    plt.figure()
    for index in range(1, feature_map_num + 1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')
        plt.axis('off')
        # imageio.imwrite("./vgg_vis_penbox/Raw/"+str(index)+".png", feature_map[index - 1])
        imageio.imwrite("./vgg_vis_penbox/LSF/"+str(index)+".png", feature_map[index - 1])
        # imageio.imwrite("./vgg_vis_penbox/OptD/"+str(index)+".png", feature_map[index - 1])
    # plt.savefig("./vgg_vis_penbox/Raw/all.png",dpi=1000)
    plt.savefig("./vgg_vis_penbox/LSF/all.png",dpi=1000)
    # plt.savefig("./vgg_vis_penbox/OptD/all.png",dpi=1000)



if __name__ == '__main__':
    # image_dir = "ILSVRC2012_val_00014429_pencil box, pencil case_org.bmp"
    image_dir = "LSF_penbox.bmp"
    # k = 2
    model = models.vgg11(pretrained=True).eval()
    use_gpu = torch.cuda.is_available()
    image_info = get_image_info(image_dir)
    
    if use_gpu:
        model = model.cuda()
        image_info = image_info.cuda()
        outputs = model(image_info)
        outputs = torch.nn.functional.softmax(outputs)
    print(outputs)
    print(outputs.argmax(1))

    # if use_gpu:
    #     model = model.cuda()
    #     image_info = image_info.cuda()
    # feature_extractor = model.features
    # feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
    # show_feature_map(feature_map)
