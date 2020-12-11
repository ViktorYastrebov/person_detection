#include "deep_sort_types.h"

namespace common {
    namespace datatypes {

        //INFO: convert to (centerx, centery, ration, h)
        DetectionBox Detection::to_xyah() const {
            DetectionBox ret = tlwh;
            ret(0, 0) += (ret(0, 2)*0.5f);
            ret(0, 1) += (ret(0, 3)*0.5f);
            ret(0, 2) /= ret(0, 3);
            return ret;
        }

        //INFO: convert to (x,y,xx,yy)
        DetectionBox Detection::to_tlbr() const {
            DetectionBox ret = tlwh;
            ret(0, 0) += ret(0, 2);
            ret(0, 1) += ret(0, 3);
            return ret;
        }

    }
}