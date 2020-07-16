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

        # self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

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

    # def __call__(self, img):
    #     assert isinstance(img, np.ndarray), "type error"
    #     img = img.astype(np.float)  # /255.
    #     img = cv2.resize(img, (64, 128))
    #     img = torch.from_numpy(img).float().permute(2, 0, 1)
    #     img = self.norm(img).unsqueeze(0)
    #     with torch.no_grad():
    #         img = img.to(self.device)
    #         feature = self.net(img)
    #     return feature.cpu().numpy()

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
