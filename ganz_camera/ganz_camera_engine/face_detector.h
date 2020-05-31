#pragma once 

#include "decl_spec.h"
#include "face_data.h"

#include "sdk_context.h"

#include <queue>
#include <mutex>

#pragma warning(push)
#pragma warning(disable: 4251)

namespace sunell_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);
    }

    class GANZ_CAMERA_ENGINE_DECL FaceDetector {
    public:
        FaceDetector(unsigned int connection, const int channel, STREAM_TYPE type, PICTURE_SIZE size);
        ~FaceDetector();

        FaceDetector(const FaceDetector &) = delete;
        FaceDetector & operator = (const FaceDetector &) = delete;

        int getStreamId() const;
        FaceDataVector getLastResults();

    private:
        friend void callback_wrapper::face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);

        unsigned int connection_;

        void handle(FaceDataVector &&faces);
        int channel_;
        STREAM_TYPE stream_type_;
        int stream_id_;

        mutable std::mutex data_mutex_;
        std::queue<FaceDataVector> data_;
    };
}

#pragma warning(pop)