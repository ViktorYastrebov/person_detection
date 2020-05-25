#include "face_handler.h"
#include "sdks.h"

#include <iostream>

namespace ganz_camera {

    //using namespace nlohmann;

    //https://stackoverflow.com/questions/4271489/how-to-use-cvimdecode-if-the-contents-of-an-image-file-are-in-a-char-array

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* picture_data, void* p_obj) {
            std::cout << "face_detection_handler call" << std::endl;
            if (p_result) {
                ganz_camera::FaceHandler* owner = static_cast<ganz_camera::FaceHandler*>(p_obj);
                const char *json_data_ptr = static_cast<char*>(*p_result);
                std::cout << "data:" << json_data_ptr << std::endl;
                FaceDataVector faces;
                faces.fromJsonData(json_data_ptr);
                owner->handle(std::move(faces));
            }
        }
    }

     FaceHandler::FaceHandler(StreamDataHolder &holder, Connection &owner)
         : holder_(holder)
         , owner_(owner)
     {
          stream_id_ = sdks_dev_face_detect_start(owner_.getHandle(), 1, 1, 5, callback_wrapper::face_detection_handler, this);
          if (stream_id_ <= 0) {
              throw std::exception("sdks_dev_face_detect_start has failed");
          }
     }

     FaceHandler::~FaceHandler()
     {
         if (stream_id_ > 0) {
            sdks_dev_face_detect_stop(owner_.getHandle(), stream_id_);
        }
     }

     int FaceHandler::getStreamId() const {
         return stream_id_;
     }

     void FaceHandler::handle(FaceDataVector &&faces) {
         holder_.put(std::move(faces));
     }
}