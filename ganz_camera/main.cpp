#include <iostream>
#include <string>

#include "sdk_context.h"
#include "face_handler.h"
#include "video_stream.h"

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

    //cv::startWindowThread();
    //cv::namedWindow("Display1");
    //cv::namedWindow("Display2");

    // INFO: I really not like that SDK creates thread pool and 
    //       each of StreamDataHolder need the thread itself too.!!!
    //       it's the problem on OpenCV UI as far as I can see.


    std::cout << "To close window please press Esc key" << std::endl;

    ganz_camera::SDKContext context;
    {
        constexpr int channel = 1;
        ganz_camera::StreamDataHolder dataHolder;
        ganz_camera::SDKContext::ConnectionPtr face_conn = context.buildConnection(host, user, pwd, is_ssl);
        ganz_camera::FaceHandler faceHandler(dataHolder, *face_conn, channel, ganz_camera::STREAM_TYPE::HD);

        ganz_camera::SDKContext::ConnectionPtr video_conn = context.buildConnection(host, user, pwd, is_ssl);
        ganz_camera::VideoStream video_stream(dataHolder, *video_conn, channel, ganz_camera::STREAM_TYPE::HD);

        video_stream.Start();
        dataHolder.start([&](ganz_camera::StreamDataHolder &owner,
                            const ganz_camera::FrameInfo &info, const ganz_camera::FaceDataVector& faces) -> void
        {
            cv::Mat clone = info.frame.clone();
            for (const auto &face : faces.faces_data_) {
                cv::Rect face_rect(face.x, face.y, face.width, face.height);
                cv::rectangle(clone, face_rect, cv::Scalar(0, 0, 255));
                std::cout << "Face detected :" << std::endl;
                std::cout << "Face Temperature: " << face.temperature << std::endl;
            }
            cv::imshow("Display1", clone);
            int key = cv::waitKey(1);
            if (key == 27) {
                // ORDER IS IMPORTANT
                video_stream.Stop();
                owner.stop();
            }
        });
        

        //constexpr int channel2 = 2;
        //ganz_camera::StreamDataHolder dataHolder2;
        //ganz_camera::SDKContext::ConnectionPtr video_conn2 = context.buildConnection(host, user, pwd, is_ssl);
        //ganz_camera::VideoStream video_stream2(dataHolder2, *video_conn2, channel2, ganz_camera::STREAM_TYPE::HD);

        //video_stream2.Start();
        //dataHolder2.start([&](ganz_camera::StreamDataHolder &owner,
        //    const ganz_camera::FrameInfo &info, const ganz_camera::FaceDataVector& faces) -> void
        //{
        //    cv::Mat clone = info.frame.clone();
        //    for (const auto &face : faces.faces_data_) {
        //        cv::Rect face_rect(face.x, face.y, face.width, face.height);
        //        cv::rectangle(clone, face_rect, cv::Scalar(0, 0, 255));
        //    }
        //    std::cout << "Display2 processed" << std::endl;
        //    //cv::imshow("Display2", clone);
        //    //int key = cv::waitKey(1);
        //    //if (key == 27) {
        //    //    owner.stop();
        //    //    video_stream2.Stop();
        //    //}
        //});

        std::cout << "Please type something" << std::endl;
        std::string input;
        std::getline(std::cin, input);
        // INFO: if we reash here wihtout video_stream.stop() &  owner.stop() called it brings to crash.
        // It can be handled simple: rewrite the stop with checking of valid handle

        //cv::destroyAllWindows();
    }
    return 0;
}