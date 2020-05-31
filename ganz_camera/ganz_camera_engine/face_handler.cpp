#include "face_handler.h"
#include "sdks.h"

#include "sdk_face.h"

namespace ganz_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* picture_data, void* p_obj) {
            if (p_result) {
                ganz_camera::FaceHandler* owner = static_cast<ganz_camera::FaceHandler*>(p_obj);
                const char *json_data_ptr = static_cast<char*>(*p_result);
                FaceDataVector faces;
                faces.fromJsonData(json_data_ptr);
                owner->handle(std::move(faces));
            }
        }
    }

     FaceHandler::FaceHandler(Connection &owner, const int channel, STREAM_TYPE type, PICTURE_SIZE size)
         : owner_(owner)
         , channel_(channel)
         , stream_type_(type)
     {
          stream_id_ = sdks_dev_face_detect_start(owner_.getHandle(), channel_, stream_type_, size, callback_wrapper::face_detection_handler, this);
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

     FaceDataVector FaceHandler::getLastResults() {
         FaceDataVector ret;
         {
             std::lock_guard<std::mutex> lock(data_mutex_);
             if (!data_.empty()) {
                 ret = data_.front();
                 data_.pop();
             }
         }
         return ret;
     }

     void FaceHandler::handle(FaceDataVector &&faces) {
         std::lock_guard<std::mutex> lock(data_mutex_);
         data_.push(std::move(faces));
     }
}