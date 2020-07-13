import torch
import argparse
from pathlib import Path
from deep_sort_pytorch.model import Net

if __name__ == '__main__':

    # base_dir = Path("d:/viktor_project/yolo_python/yolo_v5/weights")
    base_dir = Path("d:/viktor_project/person_detection/pedestrian_detection/models/deep_sort_2")
    models = ["yolov5m.pt", "yolov5l.pt", "yolov5x.pt", "yolov3-spp.pt", "ckpt.t7"]
    model_path = base_dir / models[4]

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(model_path), help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # x = torch.randn(4, 3, 128, 64)

    # Input
    # img = torch.zeros((opt.batch_size, 3, ))  # image size(1,3,320,192) iDetection
    # WITHOUT BATCHING
    img = torch.zeros(4, 3, 128, 64)

    model = Net(reid=True)
    state_dict = torch.load(str(model_path))['net_dict']
    model.load_state_dict(state_dict)

    # x = torch.randn(4, 3, 128, 64)
    # y = net(x)
    # torch.onnx.export(net, x, "deep_sort.onnx")

    # Load PyTorch model
    # model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.eval()
    # model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        # f = opt.weights.replace('.pt', '.torchscript')
        f = model_path.parent / (model_path.stem + ".torchscipt")
        ts = torch.jit.trace(model, img)
        ts.save(str(f))
        print('TorchScript export success, saved as %s' % f)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        # DEEP SORT DOES NOT HAVE FUSE FUNCTIONALITY
        has_fuse = getattr(model, "fuse", None)
        if has_fuse is not None:
            model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='images', shape=img.shape, scale=1/255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')