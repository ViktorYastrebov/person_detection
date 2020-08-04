#pragma once 

namespace common {
	struct InferDeleter {
		template <typename T>
		void operator()(T* obj) const {
			if (obj) {
				obj->destroy();
            }
        }
    };
}