#include "utils.h"
#include <opencv2/opencv.hpp>

#include <opencv2/videoio/registry.hpp>


void single_image_batching(const std::string &file) {
    const std::string url = "127.0.0.1:8001";
    const std::string model_name = "yolov5mB8";

    try {
        const int BATCH_SIZE = 4;
        std::vector<int> classes { 0 };
        cv::Mat img = cv::imread(file);
        if(img.empty()) {
            std::cout << "can't read img" << std::endl;
            return;
        }
        std::vector<cv::Mat> imgs;
        for( int i = 0; i < BATCH_SIZE; ++i) {
            imgs.push_back(img.clone());
        }

        triton_inference::GRPCClient client(url, model_name, BATCH_SIZE, classes);


        auto results = client.inference(imgs, 0.3f, 0.5f);

        client.print_stats();

        int idx = 0;
        for( const auto & detections: results) {
            for(const auto &detection : detections) {
                cv::Rect rect {
                    static_cast<int>(detection.bbox(0)),
                    static_cast<int>(detection.bbox(1)),
                    static_cast<int>(detection.bbox(2)),
                    static_cast<int>(detection.bbox(3))
                };
                cv::rectangle(imgs[idx], rect, cv::Scalar(0, 0, 255), 2);
            }
            std::string file_name = "output_" + std::to_string(idx) + ".png";
            cv::imwrite(file_name, imgs[idx]);
            ++idx;
        }
    } catch ( const std::exception &ex) {
        std::cout << " Error occurs :" << ex.what() << std::endl;
    }
}


void video_processing(const std::string &video_file) {
    const std::string url = "127.0.0.1:8001";
    const std::string model_name = "yolov5mB8";
    const int BATCH_SIZE = 4;
    std::vector<cv::Mat> imgs(BATCH_SIZE);

    try {

        auto backends = cv::videoio_registry::getBackends();
        for(const auto &backend: backends) {
            std::cout << "Avaliable : " << static_cast<int>(backend) << std::endl;
        }

        //cv::VideoCapture video_stream(video_file);
        cv::VideoCapture video_stream;
        if(!video_stream.open(video_file)) {
            std::cout << "VideoCapture open failed" << std::endl;
            return;
        }

        std::vector<int> classes {0};
        triton_inference::GRPCClient client(url, model_name, BATCH_SIZE, classes);

        cv::Mat frame;
        while (video_stream.read(frame)) {
            for(int i =0; i < BATCH_SIZE; ++i) {
                imgs[0] = frame.clone();
            }
            auto results = client.inference(imgs, 0.3f, 0.5f);
            int idx = 0;
            for( const auto & detections: results) {
                for(const auto &detection : detections) {
                    cv::Rect rect {
                        static_cast<int>(detection.bbox(0)),
                        static_cast<int>(detection.bbox(1)),
                        static_cast<int>(detection.bbox(2)),
                        static_cast<int>(detection.bbox(3))
                    };
                    cv::rectangle(imgs[idx], rect, cv::Scalar(0, 0, 255), 2);
                }
                std::string file_name = "output_" + std::to_string(idx) + ".png";
                cv::imshow(file_name, imgs[idx]);
                ++idx;
            }
        }

    } catch(const std::exception &ex) {
        std::cout << " Error occurs :" << ex.what() << std::endl;
    }
}


int main(int argc, char *argv[]) {

    //INFO: gRPC runs by default on 8001 port
    if(argc < 2) {
        std::cout << "usage : client image_file.png" << std::endl;
        return 0;
    }
    std::string file(argv[1]);
    video_processing(file);
    return 0;
}
