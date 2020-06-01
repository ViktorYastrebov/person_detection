#include "face_detector.h"
#include "sdks.h"
#include "sdk_face.h"
#include <string>

namespace sunell_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* picture_data, void* p_obj) {
            if (p_result) {
                sunell_camera::FaceDetector* owner = static_cast<sunell_camera::FaceDetector*>(p_obj);
                const char *json_data_ptr = static_cast<char*>(*p_result);
                FaceDataVector faces;
                faces.fromJsonData(json_data_ptr);
                owner->handle(std::move(faces));
            }
        }
    }

     FaceDetector::FaceDetector(unsigned int connection, const int channel, STREAM_TYPE type, PICTURE_SIZE size)
         : connection_(connection)
         , channel_(channel)
         , stream_type_(type)
     {
          stream_id_ = sdks_dev_face_detect_start(connection_, channel_, stream_type_, size, callback_wrapper::face_detection_handler, this);
          if (stream_id_ <= 0) {
              throw std::exception("sdks_dev_face_detect_start has failed");
          }
     }

     FaceDetector::~FaceDetector()
     {
         if (stream_id_ > 0) {
            sdks_dev_face_detect_stop(connection_, stream_id_);
         }
         if (connection_) {
             sdks_dev_conn_close(connection_);
         }
     }

     int FaceDetector::getStreamId() const {
         return stream_id_;
     }

     FaceDataVector FaceDetector::getLastResults() {
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

     void FaceDetector::handle(FaceDataVector &&faces) {
         std::lock_guard<std::mutex> lock(data_mutex_);
         data_.push(std::move(faces));
     }
}