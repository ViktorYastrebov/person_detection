#include <iostream>
#include <string>

#include "json/json.hpp"

#include "camera_client/sdk_context.h"
#include "camera_client/face_handler.h"
#include "camera_client/video_stream_opencv.h"
#include "camera_client/video_stream.h"
#include "sdks.h"

#include <opencv2/opencv.hpp>


//INFO: PROOF OF CONCEPT
//      ARCHITECUAL IDEA IS: SINGLE CONNECTION CAN HAVE SINGLE EVENT HANDLER !!!!
//      Current OOP design might need slitly refactorying like:
//                Connection pool
//                Each handler should create connection inside to garantee the uniqueness
//      In these conditions need to add the DataHolder with sync access( queue ) as entry point for output
//       UI should get the data only from that DataHolder


//void disconnect_handler(unsigned int handle, void* p_obj)
//{
//    std::cout << "disconnected ... " << std::endl;
//}
//
//void stream_handler(unsigned int handle, int stream_id, void* p_data, void* p_obj)
//{
//    std::cout << "stream handler " << std::endl;
//}
//
//void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* picture_data, void* p_obj)
//{
//    std::cout << "face handler" << std::endl;
//}

//void simplified_example(const std::string &host, const std::string &user, const std::string &pwd, bool is_ssl = false) {
//    sdks_dev_init(nullptr);
//    unsigned short port = 30001;
//    if (is_ssl) {
//        port = 20001;
//    }
//
//    unsigned int conn = sdks_dev_conn(host.c_str(),
//        port,
//        user.c_str(),
//        pwd.c_str(),
//        disconnect_handler, nullptr);
//
//    unsigned int conn2 = sdks_dev_conn(host.c_str(),
//        port,
//        user.c_str(),
//        pwd.c_str(),
//        disconnect_handler, nullptr);
//
//    int stream_id_ = sdks_dev_live_start(conn, 1, 1, stream_handler, nullptr);
//
//    int face_stream_id_ = sdks_dev_face_detect_start(conn2, 1, 1, 5, face_detection_handler, nullptr);
//
//    std::string input;
//    while (std::getline(std::cin, input)) {
//        if (input == "exit") {
//            break;
//        }
//    }
//    sdks_dev_face_detect_stop(conn2, face_stream_id_);
//
//    sdks_dev_live_stop(conn, stream_id_);
//
//    sdks_dev_conn_close(conn);
//    sdks_dev_quit();
//}


int main(int argc, char *argv[])
{
    std::string host;
    unsigned short port = 0;
    std::string user;
    std::string pwd;
    bool is_ssl = false;

    if (argc != 5) {
        std::cout << "Usage: server_demo.exe ssl/no-ssl IP user password";
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
        ganz_camera::StreamDataHolder dataHolder;
        ganz_camera::SDKContext::ConnectionPtr face_conn = context.buildConnection(host, user, pwd, is_ssl);
        ganz_camera::FaceHandler faceHandler(dataHolder, *face_conn);

        //std::string url = "rtsp://" + user + ":" + pwd + "@" + host + ":9554/snl/live/1/1";
        //ganz_camera::SimpleVideoStream video_stream(dataHolder, url);

        ganz_camera::SDKContext::ConnectionPtr video_conn = context.buildConnection(host, user, pwd, is_ssl);
        ganz_camera::VideoStream video_stream(dataHolder, *video_conn, 1, ganz_camera::VideoStream::HD);

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