#include "face_handler.h"
#include "sdks.h"



namespace ganz_camera {

    using namespace nlohmann;

    //https://stackoverflow.com/questions/4271489/how-to-use-cvimdecode-if-the-contents-of-an-image-file-are-in-a-char-array

    namespace callback_wrapper {
        void face_detection_handler(unsigned int handle, int stream_id, void** p_result, void* picture_data, void* p_obj) {
            ganz_camera::FaceHandler* owner = static_cast<ganz_camera::FaceHandler*>(p_obj);
            if (p_result) {
                //INFO: need deeper investigation with localization strings, may be need to use ICU
                const char *json_data_ptr = static_cast<char*>(*p_result);
                std::string json_string(json_data_ptr);
                auto json_data = json::parse(json_string);

                const char *picture_ptr = static_cast<char*>(picture_data);
                owner->handle(json_data, picture_ptr);
            }
        }
    }

     FaceHandler::FaceHandler(Connection &owner)
         : owner_(owner)
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

     void FaceHandler::handle(const nlohmann::json &json, const char *picture) {
         auto image_size = json["PictureLen"].get<int>();
         //Add cv Mat Here
         auto detected_list = json["TargetDetectList"];
         for (const auto &entry : detected_list) {
             auto x = entry["X"].get<int>();
             auto y = entry["Y"].get<int>();
             auto w = entry["W"].get<int>();
             auto h = entry["H"].get<int>();
             auto faceAttributes = entry["PersonFace"];
             auto confidence = faceAttributes["Confidence"].get<int>();
             auto temperature = faceAttributes["Temperature"].get<double>();
         }
     }
}