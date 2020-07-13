import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path)['net_dict']
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # for i in range(num_batches):
    #     s, e = i * batch_size, (i + 1) * batch_size
    #     batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
    #     out[s:e] = f(batch_data_dict)
    # if e < len(out):
    #     batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
    #     out[e:] = f(batch_data_dict)

    # def __call__(self, imgs: list, batching=64):
    #     assert isinstance(imgs[0], np.ndarray), "type error"
    #     num_batches = int(len(imgs) / batching)
    #     for i in range(num_batches):

    def __call__(self, img):
        assert isinstance(img, np.ndarray), "type error"
        img = img.astype(np.float)  # /255.
        img = cv2.resize(img, (64, 128))
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        img = self.norm(img).unsqueeze(0)
        with torch.no_grad():
            img = img.to(self.device)
            feature = self.net(img)
        return feature.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
