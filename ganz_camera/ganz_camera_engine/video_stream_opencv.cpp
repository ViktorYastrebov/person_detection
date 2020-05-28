#include "camera_client/video_stream_opencv.h"

namespace ganz_camera {

    SimpleVideoStream::SimpleVideoStream(StreamDataHolder &holder, const std::string &url)
        :BaseVideoStream(holder)
        , url_(url)
        , stop_flag_(false)
    {}

    void SimpleVideoStream::process() {
        cv::Mat frame;
        while (stop_flag_) {
            if (stream_.read(frame)) {
                //INFO: OpenCV, cvRetrieveFrame() :
                // return image stored inside the video capturing structure.
                // It is not allowed to modify or release the image!
                // You can copy the frame using cvCloneImage and then do whatever you want with the copy.
                handleFrame(frame.clone());
            }
        }
    }

    void SimpleVideoStream::Start() {

        stream_ = cv::VideoCapture(url_);
        if (!stream_.isOpened()) {
            throw std::exception("Can't open video stream");
        }
        stop_flag_ = true;
        auto proc_func = std::bind(&SimpleVideoStream::process, this);
        runner_ = std::thread(proc_func);
    }

    void SimpleVideoStream::Stop() {
        stop_flag_ = false;
        if (runner_.joinable()) {
            runner_.join();
        }
        stream_.release();
    }

}