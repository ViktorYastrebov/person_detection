import torch
import struct
import argparse
from pathlib import Path
from utils.torch_utils import select_device

def convert(model_str):
    # Initialize
    device = select_device('cpu')
    # Load model
    model_path = Path(model_str)
    model = torch.load(str(model_path), map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    out_name = model_path.stem + ".wts"
    f = open(out_name, 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="One of the model YoloV5s/YoloV5m/YoloV5m")
    args = parser.parse_args()
    if args.model is not None:
        convert(args.model)
    else:
        print(f"Usage python gen_wts.py --model file")



