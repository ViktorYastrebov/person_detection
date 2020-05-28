#include "face_handler.h"
#include "sdks.h"

#include "sdk_face.h"

#include <iostream>

namespace ganz_camera {

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* picture_data, void* p_obj) {
            //std::cout << "face_detection_handler call" << std::endl;
            if (p_result) {
                ganz_camera::FaceHandler* owner = static_cast<ganz_camera::FaceHandler*>(p_obj);
                const char *json_data_ptr = static_cast<char*>(*p_result);
                std::cout << "data:" << json_data_ptr << std::endl;
                FaceDataVector faces;
                faces.fromJsonData(json_data_ptr);
                owner->handle(std::move(faces));
            }
        }

        void face_callback(unsigned int handle, int pic_type, void* p_data, int *data_len, void** p_result, void* p_obj) {
            std::cout << "Face callback 2" << std::endl;
        }

    }

     FaceHandler::FaceHandler(StreamDataHolder &holder, Connection &owner, const int channel, STREAM_TYPE type)
         : holder_(holder)
         , owner_(owner)
         , channel_(channel)
         , stream_type_(type)
     {

         char *ptr = 0;
         int ret = sdks_get_face_detect_param(owner_.getHandle(), channel_, &ptr);
         if (!ret) {
             std::string result(ptr);
             std::cout << "Params: " << result << std::endl;
             sdks_free_result(ptr);
         }

         // 4: Small picture. 5: Big picture. 7: size chart
          stream_id_ = sdks_dev_face_detect_start(owner_.getHandle(), channel_, stream_type_, 5, callback_wrapper::face_detection_handler, this);
          if (stream_id_ <= 0) {
              throw std::exception("sdks_dev_face_detect_start has failed");
          }
          //start_face = sdks_start_face(owner_.getHandle(), callback_wrapper::face_callback, this);
     }

     FaceHandler::~FaceHandler()
     {
         //if (!start_face) {
         //    sdks_stop_face(owner_.getHandle());
         //}

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