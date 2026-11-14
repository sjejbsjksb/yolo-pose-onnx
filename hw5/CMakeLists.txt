cmake_minimum_required(VERSION 3.16)
project(ArmorPnP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 强制 Release：Debug 下 OpenCV 的 CV_DbgAssert 全开，帧率会明显下降
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "已强制 Release 构建（Debug 会明显拉低帧率）")
endif()

add_compile_options(-Wall -Wextra -O2)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ---- 海康 MVS SDK ----
set(MVS_ROOT "/opt/MVS" CACHE PATH "Hikvision MVS SDK root")
find_path(MVS_INCLUDE_DIR NAMES MvCameraControl.h
          HINTS "${MVS_ROOT}/include" "/opt/MVS/include")
find_library(MVS_LIBRARY NAMES MvCameraControl
             HINTS "${MVS_ROOT}/lib/64" "/opt/MVS/lib/64" "/opt/MVS/lib")

if(NOT MVS_INCLUDE_DIR)
  message(FATAL_ERROR "找不到 MvCameraControl.h，确认 MVS 装在 /opt/MVS")
endif()
if(NOT MVS_LIBRARY)
  message(FATAL_ERROR "找不到 libMvCameraControl.so，确认 /opt/MVS/lib/64 下有该文件")
endif()

message(STATUS "OpenCV      : ${OpenCV_VERSION}")
message(STATUS "MVS include : ${MVS_INCLUDE_DIR}")
message(STATUS "MVS library : ${MVS_LIBRARY}")

function(setup_target tgt)
  target_include_directories(${tgt} PRIVATE ${OpenCV_INCLUDE_DIRS} ${MVS_INCLUDE_DIR})
  target_link_libraries(${tgt} PRIVATE ${OpenCV_LIBS} ${MVS_LIBRARY} Threads::Threads)
  set_target_properties(${tgt} PROPERTIES
      BUILD_RPATH "/opt/MVS/lib/64"
      INSTALL_RPATH "/opt/MVS/lib/64")
endfunction()

# 任务 1：相机标定
add_executable(Calibrate calibrate.cpp)
setup_target(Calibrate)

# 任务 2~4：识别 + PnP + 卡尔曼
add_executable(ArmorPnP main.cpp)
setup_target(ArmorPnP)
