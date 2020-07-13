#include <torch/torch.h>
#include <iostream>


//std::unique_ptr<torch::jit::Module> load()

int main() {

    try {
        if (torch::cuda::is_available()) {
            torch::Device gpu_device("cuda:0");

            const std::string path = "d:/viktor_project/person_detection/pytorch_demos/models/yolov3-spp.pt";
            //torch::jit::Module model = torch::jit::load(path);
            //torch::nn::Module model;
            std::vector<torch::Tensor> tensors;
            torch::load(tensors, "d:/viktor_project/person_detection/pytorch_demos/models/yolov3-spp.pt");
            //model.to(gpu_device);
            //model.eval();
            //torch::Tensor tensor = torch::eye(3);
            //std::cout << tensor << std::endl;
        } else {
            std::cout << "GPU is not avaliable" << std::endl;
        }
    } catch (const std::exception &ex) {
        std::cout << "Error occured: " << ex.what() << std::endl;
    }
    catch (...) {
        std::cout << "Unhandled exception" << std::endl;
    }
    return 0;
}