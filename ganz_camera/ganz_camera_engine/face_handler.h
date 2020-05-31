#pragma once 

#include "decl_spec.h"
#include "connection.h"
#include "face_data.h"

#include <queue>
#include <mutex>

#pragma warning(push)
#pragma warning(disable: 4251)

namespace ganz_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);
    }

    class GANZ_CAMERA_ENGINE_DECL FaceHandler {
    public:

        enum PICTURE_SIZE: int {
            SMALL = 4,
            BIG = 5,
            SIZE_CHART = 7 // NEEDS TO CHECK
        };

        FaceHandler(Connection &owner, const int channel, STREAM_TYPE type, PICTURE_SIZE size);
        ~FaceHandler();

        FaceHandler(const FaceHandler &) = delete;
        FaceHandler & operator = (const FaceHandler &) = delete;

        int getStreamId() const;
        FaceDataVector getLastResults();

    private:
        friend void callback_wrapper::face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* p_data, void* p_obj);

        void handle(FaceDataVector &&faces);
        Connection &owner_;
        int channel_;
        STREAM_TYPE stream_type_;
        int stream_id_;

        mutable std::mutex data_mutex_;
        std::queue<FaceDataVector> data_;
    };
}

#pragma warning(pop)