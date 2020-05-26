#include <iostream>
#include <string>

#include "json/json.hpp"

#include "camera_client/sdk_context.h"
#include "camera_client/face_handler.h"
#include "camera_client/video_stream_opencv.h"
#include "camera_client/video_stream.h"
#include "sdks.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    std::string host;
    unsigned short port = 0;
    std::string user;
    std::string pwd;
    std::string str_port;
    bool is_ssl = false;

    if (argc != 5) {
        std::cout << "Usage: server_demo.exe ssl/no-ssl IP port user password";
        return 0;
    } else {
        std::string test_ssl(argv[1]);
        is_ssl = test_ssl == "ssl";
        host = std::string(argv[2]);
        user = std::string(argv[3]);
        pwd = std::string(argv[4]);
    }

    ganz_camera::SDKContext context;
    {
        const int channel = 1;
        ganz_camera::StreamDataHolder dataHolder;
        ganz_camera::SDKContext::ConnectionPtr face_conn = context.buildConnection(host, user, pwd, is_ssl);
        ganz_camera::FaceHandler faceHandler(dataHolder, *face_conn, channel, ganz_camera::STREAM_TYPE::HD);

        ganz_camera::SDKContext::ConnectionPtr video_conn = context.buildConnection(host, user, pwd, is_ssl);
        ganz_camera::VideoStream video_stream(dataHolder, *video_conn, channel, ganz_camera::STREAM_TYPE::HD);

        cv::startWindowThread();
        cv::namedWindow("Display");

        video_stream.Start();
        dataHolder.start([](ganz_camera::StreamDataHolder &owner,
                            cv::Mat data, const ganz_camera::FaceDataVector& faces) -> void
        {
            cv::Mat clone = data.clone();
            for (const auto &face : faces.faces_data_) {
                cv::Rect face_rect(face.x, face.y, face.width, face.height);
                cv::rectangle(clone, face_rect, cv::Scalar(0, 0, 255));
            }
            cv::imshow("Display", clone);
            int key = cv::waitKey(1);
            if (key == 27) {
                owner.stop();
            }
        });
        video_stream.Stop();
        cv::destroyAllWindows();
    }
    return 0;
}