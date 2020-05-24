Dependencies:
	1. OpenCV built with ffmpeg support( -DWITH_FFMPEG=ON option)
	2. TBB library, you can find it in the deps folder
	3. Native Ganz camera SDK for direct connection

	
Project desciption:
	Its build in the OOP style based on the Ganz Camera SDK interface. 
	It has been figured out that SDK uses thread pool(can see from output) and used Event base architecute(callbacks).
	For syncronize the data from different stream is used tbb::concurent_queue inside the StreamDataHolder
	`StreamDataHolder::Start` method is the entry point of processing data.
	
Build & Run:
	run build_debug.bat or build.bat correspondely to Debug or Release build.(It's tested under the debug mode only)
	Output you may find in the build folder where all dependecy DLL must be copied.
	
	Example of running: "no-ssl" "IP" "user" "password"
	
	

