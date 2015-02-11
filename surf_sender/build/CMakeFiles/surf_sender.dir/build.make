# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gf/Projects/opencv/cuda_find/surf_sender

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gf/Projects/opencv/cuda_find/surf_sender/build

# Include any dependencies generated for this target.
include CMakeFiles/surf_sender.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/surf_sender.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/surf_sender.dir/flags.make

CMakeFiles/surf_sender.dir/surf_sender.cpp.o: CMakeFiles/surf_sender.dir/flags.make
CMakeFiles/surf_sender.dir/surf_sender.cpp.o: ../surf_sender.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/gf/Projects/opencv/cuda_find/surf_sender/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/surf_sender.dir/surf_sender.cpp.o"
	/usr/local/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/surf_sender.dir/surf_sender.cpp.o -c /home/gf/Projects/opencv/cuda_find/surf_sender/surf_sender.cpp

CMakeFiles/surf_sender.dir/surf_sender.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surf_sender.dir/surf_sender.cpp.i"
	/usr/local/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/gf/Projects/opencv/cuda_find/surf_sender/surf_sender.cpp > CMakeFiles/surf_sender.dir/surf_sender.cpp.i

CMakeFiles/surf_sender.dir/surf_sender.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surf_sender.dir/surf_sender.cpp.s"
	/usr/local/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/gf/Projects/opencv/cuda_find/surf_sender/surf_sender.cpp -o CMakeFiles/surf_sender.dir/surf_sender.cpp.s

CMakeFiles/surf_sender.dir/surf_sender.cpp.o.requires:
.PHONY : CMakeFiles/surf_sender.dir/surf_sender.cpp.o.requires

CMakeFiles/surf_sender.dir/surf_sender.cpp.o.provides: CMakeFiles/surf_sender.dir/surf_sender.cpp.o.requires
	$(MAKE) -f CMakeFiles/surf_sender.dir/build.make CMakeFiles/surf_sender.dir/surf_sender.cpp.o.provides.build
.PHONY : CMakeFiles/surf_sender.dir/surf_sender.cpp.o.provides

CMakeFiles/surf_sender.dir/surf_sender.cpp.o.provides.build: CMakeFiles/surf_sender.dir/surf_sender.cpp.o

CMakeFiles/surf_sender.dir/data_writer.cpp.o: CMakeFiles/surf_sender.dir/flags.make
CMakeFiles/surf_sender.dir/data_writer.cpp.o: ../data_writer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/gf/Projects/opencv/cuda_find/surf_sender/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/surf_sender.dir/data_writer.cpp.o"
	/usr/local/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/surf_sender.dir/data_writer.cpp.o -c /home/gf/Projects/opencv/cuda_find/surf_sender/data_writer.cpp

CMakeFiles/surf_sender.dir/data_writer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surf_sender.dir/data_writer.cpp.i"
	/usr/local/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/gf/Projects/opencv/cuda_find/surf_sender/data_writer.cpp > CMakeFiles/surf_sender.dir/data_writer.cpp.i

CMakeFiles/surf_sender.dir/data_writer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surf_sender.dir/data_writer.cpp.s"
	/usr/local/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/gf/Projects/opencv/cuda_find/surf_sender/data_writer.cpp -o CMakeFiles/surf_sender.dir/data_writer.cpp.s

CMakeFiles/surf_sender.dir/data_writer.cpp.o.requires:
.PHONY : CMakeFiles/surf_sender.dir/data_writer.cpp.o.requires

CMakeFiles/surf_sender.dir/data_writer.cpp.o.provides: CMakeFiles/surf_sender.dir/data_writer.cpp.o.requires
	$(MAKE) -f CMakeFiles/surf_sender.dir/build.make CMakeFiles/surf_sender.dir/data_writer.cpp.o.provides.build
.PHONY : CMakeFiles/surf_sender.dir/data_writer.cpp.o.provides

CMakeFiles/surf_sender.dir/data_writer.cpp.o.provides.build: CMakeFiles/surf_sender.dir/data_writer.cpp.o

# Object files for target surf_sender
surf_sender_OBJECTS = \
"CMakeFiles/surf_sender.dir/surf_sender.cpp.o" \
"CMakeFiles/surf_sender.dir/data_writer.cpp.o"

# External object files for target surf_sender
surf_sender_EXTERNAL_OBJECTS =

surf_sender: CMakeFiles/surf_sender.dir/surf_sender.cpp.o
surf_sender: CMakeFiles/surf_sender.dir/data_writer.cpp.o
surf_sender: CMakeFiles/surf_sender.dir/build.make
surf_sender: /usr/local/cuda/lib64/libcudart.so
surf_sender: /usr/local/lib/libopencv_videostab.so.2.4.10
surf_sender: /usr/local/lib/libopencv_video.so.2.4.10
surf_sender: /usr/local/lib/libopencv_ts.a
surf_sender: /usr/local/lib/libopencv_superres.so.2.4.10
surf_sender: /usr/local/lib/libopencv_stitching.so.2.4.10
surf_sender: /usr/local/lib/libopencv_photo.so.2.4.10
surf_sender: /usr/local/lib/libopencv_ocl.so.2.4.10
surf_sender: /usr/local/lib/libopencv_objdetect.so.2.4.10
surf_sender: /usr/local/lib/libopencv_nonfree.so.2.4.10
surf_sender: /usr/local/lib/libopencv_ml.so.2.4.10
surf_sender: /usr/local/lib/libopencv_legacy.so.2.4.10
surf_sender: /usr/local/lib/libopencv_imgproc.so.2.4.10
surf_sender: /usr/local/lib/libopencv_highgui.so.2.4.10
surf_sender: /usr/local/lib/libopencv_gpu.so.2.4.10
surf_sender: /usr/local/lib/libopencv_flann.so.2.4.10
surf_sender: /usr/local/lib/libopencv_features2d.so.2.4.10
surf_sender: /usr/local/lib/libopencv_core.so.2.4.10
surf_sender: /usr/local/lib/libopencv_contrib.so.2.4.10
surf_sender: /usr/local/lib/libopencv_calib3d.so.2.4.10
surf_sender: /home/gf/Libraries/libfreenect2/examples/protonect/lib/libfreenect2.so
surf_sender: /home/gf/Libraries/libfreenect2/depends/glew/lib64/libGLEW.so
surf_sender: /home/gf/Libraries/libfreenect2/examples/protonect/lib/libglfw.so.3.0
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_system.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_thread.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_mpi.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
surf_sender: /usr/lib/x86_64-linux-gnu/libpthread.so
surf_sender: /usr/local/lib/libpcl_common.so
surf_sender: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
surf_sender: /usr/local/lib/libpcl_kdtree.so
surf_sender: /usr/local/lib/libpcl_octree.so
surf_sender: /usr/local/lib/libpcl_search.so
surf_sender: /usr/local/lib/libpcl_sample_consensus.so
surf_sender: /usr/local/lib/libpcl_filters.so
surf_sender: /usr/lib/libOpenNI.so
surf_sender: /usr/local/lib/libOpenNI2.so
surf_sender: /usr/lib/libvtkCommon.so.5.8.0
surf_sender: /usr/lib/libvtkRendering.so.5.8.0
surf_sender: /usr/lib/libvtkHybrid.so.5.8.0
surf_sender: /usr/lib/libvtkCharts.so.5.8.0
surf_sender: /usr/local/lib/libpcl_io.so
surf_sender: /usr/local/lib/libpcl_features.so
surf_sender: /usr/local/lib/libpcl_keypoints.so
surf_sender: /usr/local/lib/libpcl_gpu_containers.so
surf_sender: /usr/local/lib/libpcl_gpu_utils.so
surf_sender: /usr/local/lib/libpcl_gpu_octree.so
surf_sender: /usr/local/lib/libpcl_gpu_features.so
surf_sender: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
surf_sender: /usr/local/lib/libpcl_gpu_segmentation.so
surf_sender: /usr/local/lib/libpcl_gpu_kinfu.so
surf_sender: /usr/local/lib/libpcl_visualization.so
surf_sender: /usr/local/lib/libpcl_cuda_features.so
surf_sender: /usr/local/lib/libpcl_cuda_io.so
surf_sender: /usr/local/lib/libpcl_cuda_sample_consensus.so
surf_sender: /usr/local/lib/libpcl_cuda_segmentation.so
surf_sender: /usr/local/lib/libpcl_ml.so
surf_sender: /usr/local/lib/libpcl_segmentation.so
surf_sender: /usr/lib/x86_64-linux-gnu/libqhull.so
surf_sender: /usr/local/lib/libpcl_surface.so
surf_sender: /usr/local/lib/libpcl_registration.so
surf_sender: /usr/local/lib/libpcl_recognition.so
surf_sender: /usr/local/lib/libpcl_outofcore.so
surf_sender: /usr/local/lib/libpcl_stereo.so
surf_sender: /usr/local/lib/libpcl_people.so
surf_sender: /usr/local/lib/libpcl_tracking.so
surf_sender: /usr/local/lib/libpcl_apps.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_system.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_thread.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_mpi.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
surf_sender: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
surf_sender: /usr/lib/x86_64-linux-gnu/libpthread.so
surf_sender: /usr/lib/x86_64-linux-gnu/libqhull.so
surf_sender: /usr/lib/libOpenNI.so
surf_sender: /usr/local/lib/libOpenNI2.so
surf_sender: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
surf_sender: /usr/lib/libvtkCommon.so.5.8.0
surf_sender: /usr/lib/libvtkRendering.so.5.8.0
surf_sender: /usr/lib/libvtkHybrid.so.5.8.0
surf_sender: /usr/lib/libvtkCharts.so.5.8.0
surf_sender: /usr/local/lib/libjsoncpp.a
surf_sender: /usr/local/lib/libpcl_common.so
surf_sender: /usr/local/lib/libpcl_kdtree.so
surf_sender: /usr/local/lib/libpcl_octree.so
surf_sender: /usr/local/lib/libpcl_search.so
surf_sender: /usr/local/lib/libpcl_sample_consensus.so
surf_sender: /usr/local/lib/libpcl_filters.so
surf_sender: /usr/local/lib/libpcl_io.so
surf_sender: /usr/local/lib/libpcl_features.so
surf_sender: /usr/local/lib/libpcl_keypoints.so
surf_sender: /usr/local/lib/libpcl_gpu_containers.so
surf_sender: /usr/local/lib/libpcl_gpu_utils.so
surf_sender: /usr/local/lib/libpcl_gpu_octree.so
surf_sender: /usr/local/lib/libpcl_gpu_features.so
surf_sender: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
surf_sender: /usr/local/lib/libpcl_gpu_segmentation.so
surf_sender: /usr/local/lib/libpcl_gpu_kinfu.so
surf_sender: /usr/local/lib/libpcl_visualization.so
surf_sender: /usr/local/lib/libpcl_cuda_features.so
surf_sender: /usr/local/lib/libpcl_cuda_io.so
surf_sender: /usr/local/lib/libpcl_cuda_sample_consensus.so
surf_sender: /usr/local/lib/libpcl_cuda_segmentation.so
surf_sender: /usr/local/lib/libpcl_ml.so
surf_sender: /usr/local/lib/libpcl_segmentation.so
surf_sender: /usr/local/lib/libpcl_surface.so
surf_sender: /usr/local/lib/libpcl_registration.so
surf_sender: /usr/local/lib/libpcl_recognition.so
surf_sender: /usr/local/lib/libpcl_outofcore.so
surf_sender: /usr/local/lib/libpcl_stereo.so
surf_sender: /usr/local/lib/libpcl_people.so
surf_sender: /usr/local/lib/libpcl_tracking.so
surf_sender: /usr/local/lib/libpcl_apps.so
surf_sender: /usr/local/lib/libjsoncpp.a
surf_sender: /usr/local/lib/libopencv_nonfree.so.2.4.10
surf_sender: /usr/local/lib/libopencv_ocl.so.2.4.10
surf_sender: /usr/local/lib/libopencv_gpu.so.2.4.10
surf_sender: /usr/local/lib/libopencv_photo.so.2.4.10
surf_sender: /usr/local/lib/libopencv_objdetect.so.2.4.10
surf_sender: /usr/local/lib/libopencv_legacy.so.2.4.10
surf_sender: /usr/local/lib/libopencv_video.so.2.4.10
surf_sender: /usr/local/lib/libopencv_ml.so.2.4.10
surf_sender: /usr/local/lib/libopencv_calib3d.so.2.4.10
surf_sender: /usr/local/lib/libopencv_features2d.so.2.4.10
surf_sender: /usr/local/lib/libopencv_highgui.so.2.4.10
surf_sender: /usr/local/lib/libopencv_imgproc.so.2.4.10
surf_sender: /usr/local/lib/libopencv_flann.so.2.4.10
surf_sender: /usr/local/lib/libopencv_core.so.2.4.10
surf_sender: /usr/lib/libvtkViews.so.5.8.0
surf_sender: /usr/lib/libvtkInfovis.so.5.8.0
surf_sender: /usr/lib/libvtkWidgets.so.5.8.0
surf_sender: /usr/lib/libvtkHybrid.so.5.8.0
surf_sender: /usr/lib/libvtkParallel.so.5.8.0
surf_sender: /usr/lib/libvtkVolumeRendering.so.5.8.0
surf_sender: /usr/lib/libvtkRendering.so.5.8.0
surf_sender: /usr/lib/libvtkGraphics.so.5.8.0
surf_sender: /usr/lib/libvtkImaging.so.5.8.0
surf_sender: /usr/lib/libvtkIO.so.5.8.0
surf_sender: /usr/lib/libvtkFiltering.so.5.8.0
surf_sender: /usr/lib/libvtkCommon.so.5.8.0
surf_sender: /usr/lib/libvtksys.so.5.8.0
surf_sender: CMakeFiles/surf_sender.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable surf_sender"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/surf_sender.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/surf_sender.dir/build: surf_sender
.PHONY : CMakeFiles/surf_sender.dir/build

CMakeFiles/surf_sender.dir/requires: CMakeFiles/surf_sender.dir/surf_sender.cpp.o.requires
CMakeFiles/surf_sender.dir/requires: CMakeFiles/surf_sender.dir/data_writer.cpp.o.requires
.PHONY : CMakeFiles/surf_sender.dir/requires

CMakeFiles/surf_sender.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/surf_sender.dir/cmake_clean.cmake
.PHONY : CMakeFiles/surf_sender.dir/clean

CMakeFiles/surf_sender.dir/depend:
	cd /home/gf/Projects/opencv/cuda_find/surf_sender/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gf/Projects/opencv/cuda_find/surf_sender /home/gf/Projects/opencv/cuda_find/surf_sender /home/gf/Projects/opencv/cuda_find/surf_sender/build /home/gf/Projects/opencv/cuda_find/surf_sender/build /home/gf/Projects/opencv/cuda_find/surf_sender/build/CMakeFiles/surf_sender.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/surf_sender.dir/depend

