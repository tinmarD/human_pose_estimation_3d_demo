# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build

# Include any dependencies generated for this target.
include CMakeFiles/pose_extractor.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pose_extractor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pose_extractor.dir/flags.make

CMakeFiles/pose_extractor.dir/wrapper.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/wrapper.o: ../wrapper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pose_extractor.dir/wrapper.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pose_extractor.dir/wrapper.o -c /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/wrapper.cpp

CMakeFiles/pose_extractor.dir/wrapper.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/wrapper.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/wrapper.cpp > CMakeFiles/pose_extractor.dir/wrapper.i

CMakeFiles/pose_extractor.dir/wrapper.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/wrapper.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/wrapper.cpp -o CMakeFiles/pose_extractor.dir/wrapper.s

CMakeFiles/pose_extractor.dir/wrapper.o.requires:

.PHONY : CMakeFiles/pose_extractor.dir/wrapper.o.requires

CMakeFiles/pose_extractor.dir/wrapper.o.provides: CMakeFiles/pose_extractor.dir/wrapper.o.requires
	$(MAKE) -f CMakeFiles/pose_extractor.dir/build.make CMakeFiles/pose_extractor.dir/wrapper.o.provides.build
.PHONY : CMakeFiles/pose_extractor.dir/wrapper.o.provides

CMakeFiles/pose_extractor.dir/wrapper.o.provides.build: CMakeFiles/pose_extractor.dir/wrapper.o


CMakeFiles/pose_extractor.dir/src/extract_poses.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/src/extract_poses.o: ../src/extract_poses.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/pose_extractor.dir/src/extract_poses.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pose_extractor.dir/src/extract_poses.o -c /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/extract_poses.cpp

CMakeFiles/pose_extractor.dir/src/extract_poses.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/src/extract_poses.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/extract_poses.cpp > CMakeFiles/pose_extractor.dir/src/extract_poses.i

CMakeFiles/pose_extractor.dir/src/extract_poses.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/src/extract_poses.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/extract_poses.cpp -o CMakeFiles/pose_extractor.dir/src/extract_poses.s

CMakeFiles/pose_extractor.dir/src/extract_poses.o.requires:

.PHONY : CMakeFiles/pose_extractor.dir/src/extract_poses.o.requires

CMakeFiles/pose_extractor.dir/src/extract_poses.o.provides: CMakeFiles/pose_extractor.dir/src/extract_poses.o.requires
	$(MAKE) -f CMakeFiles/pose_extractor.dir/build.make CMakeFiles/pose_extractor.dir/src/extract_poses.o.provides.build
.PHONY : CMakeFiles/pose_extractor.dir/src/extract_poses.o.provides

CMakeFiles/pose_extractor.dir/src/extract_poses.o.provides.build: CMakeFiles/pose_extractor.dir/src/extract_poses.o


CMakeFiles/pose_extractor.dir/src/human_pose.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/src/human_pose.o: ../src/human_pose.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/pose_extractor.dir/src/human_pose.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pose_extractor.dir/src/human_pose.o -c /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/human_pose.cpp

CMakeFiles/pose_extractor.dir/src/human_pose.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/src/human_pose.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/human_pose.cpp > CMakeFiles/pose_extractor.dir/src/human_pose.i

CMakeFiles/pose_extractor.dir/src/human_pose.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/src/human_pose.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/human_pose.cpp -o CMakeFiles/pose_extractor.dir/src/human_pose.s

CMakeFiles/pose_extractor.dir/src/human_pose.o.requires:

.PHONY : CMakeFiles/pose_extractor.dir/src/human_pose.o.requires

CMakeFiles/pose_extractor.dir/src/human_pose.o.provides: CMakeFiles/pose_extractor.dir/src/human_pose.o.requires
	$(MAKE) -f CMakeFiles/pose_extractor.dir/build.make CMakeFiles/pose_extractor.dir/src/human_pose.o.provides.build
.PHONY : CMakeFiles/pose_extractor.dir/src/human_pose.o.provides

CMakeFiles/pose_extractor.dir/src/human_pose.o.provides.build: CMakeFiles/pose_extractor.dir/src/human_pose.o


CMakeFiles/pose_extractor.dir/src/peak.o: CMakeFiles/pose_extractor.dir/flags.make
CMakeFiles/pose_extractor.dir/src/peak.o: ../src/peak.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/pose_extractor.dir/src/peak.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pose_extractor.dir/src/peak.o -c /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/peak.cpp

CMakeFiles/pose_extractor.dir/src/peak.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_extractor.dir/src/peak.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/peak.cpp > CMakeFiles/pose_extractor.dir/src/peak.i

CMakeFiles/pose_extractor.dir/src/peak.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_extractor.dir/src/peak.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/src/peak.cpp -o CMakeFiles/pose_extractor.dir/src/peak.s

CMakeFiles/pose_extractor.dir/src/peak.o.requires:

.PHONY : CMakeFiles/pose_extractor.dir/src/peak.o.requires

CMakeFiles/pose_extractor.dir/src/peak.o.provides: CMakeFiles/pose_extractor.dir/src/peak.o.requires
	$(MAKE) -f CMakeFiles/pose_extractor.dir/build.make CMakeFiles/pose_extractor.dir/src/peak.o.provides.build
.PHONY : CMakeFiles/pose_extractor.dir/src/peak.o.provides

CMakeFiles/pose_extractor.dir/src/peak.o.provides.build: CMakeFiles/pose_extractor.dir/src/peak.o


# Object files for target pose_extractor
pose_extractor_OBJECTS = \
"CMakeFiles/pose_extractor.dir/wrapper.o" \
"CMakeFiles/pose_extractor.dir/src/extract_poses.o" \
"CMakeFiles/pose_extractor.dir/src/human_pose.o" \
"CMakeFiles/pose_extractor.dir/src/peak.o"

# External object files for target pose_extractor
pose_extractor_EXTERNAL_OBJECTS =

pose_extractor.so: CMakeFiles/pose_extractor.dir/wrapper.o
pose_extractor.so: CMakeFiles/pose_extractor.dir/src/extract_poses.o
pose_extractor.so: CMakeFiles/pose_extractor.dir/src/human_pose.o
pose_extractor.so: CMakeFiles/pose_extractor.dir/src/peak.o
pose_extractor.so: CMakeFiles/pose_extractor.dir/build.make
pose_extractor.so: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_imgproc.so.4.3.0
pose_extractor.so: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_core.so.4.3.0
pose_extractor.so: CMakeFiles/pose_extractor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared module pose_extractor.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pose_extractor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pose_extractor.dir/build: pose_extractor.so

.PHONY : CMakeFiles/pose_extractor.dir/build

CMakeFiles/pose_extractor.dir/requires: CMakeFiles/pose_extractor.dir/wrapper.o.requires
CMakeFiles/pose_extractor.dir/requires: CMakeFiles/pose_extractor.dir/src/extract_poses.o.requires
CMakeFiles/pose_extractor.dir/requires: CMakeFiles/pose_extractor.dir/src/human_pose.o.requires
CMakeFiles/pose_extractor.dir/requires: CMakeFiles/pose_extractor.dir/src/peak.o.requires

.PHONY : CMakeFiles/pose_extractor.dir/requires

CMakeFiles/pose_extractor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pose_extractor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pose_extractor.dir/clean

CMakeFiles/pose_extractor.dir/depend:
	cd /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build /home/tinmar/Desktop/WiiCare/HumanPoseEstimation/code/human_pose_estimation_3d_demo_jamal/pose_extractor/build/CMakeFiles/pose_extractor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pose_extractor.dir/depend

