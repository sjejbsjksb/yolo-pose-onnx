cmake_minimum_required(VERSION 3.16)
project(YoloCamera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Qt Creator 默认传 -DCMAKE_BUILD_TYPE=Debug，Debug 构建会打开 OpenCV 的
# CV_DbgAssert 并且不定义 NDEBUG，Mat 操作明显变慢（帧率能差 30%+）。
# 这里强制拉回 Release；真要单步调试就手动改成 RelWithDebInfo。
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "已强制 Release 构建（Debug 会明显拉低帧率）")
endif()

add_compile_options(-Wall -Wextra -O2)

# ---------------------------------------------------------------------------
#  main.cpp 在根目录还是 src/ 下都能找到 —— 不用你挪文件
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
  set(MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
  set(MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
else()
  message(FATAL_ERROR
    "没找到 main.cpp！\n"
    "  应该放在 ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp\n"
    "  或者 ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
endif()
message(STATUS "source         : ${MAIN_SRC}")

# ---------------------------------------------------------------------------
#  OpenCV（要 dnn 模块）
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
#  海康 MVS SDK（Linux 默认 /opt/MVS）
# ---------------------------------------------------------------------------
set(MVS_ROOT "/opt/MVS" CACHE PATH "Hikvision MVS SDK root")

find_path(MVS_INCLUDE_DIR
          NAMES MvCameraControl.h
          HINTS "${MVS_ROOT}/include" "/opt/MVS/include")

find_library(MVS_LIBRARY
             NAMES MvCameraControl
             HINTS "${MVS_ROOT}/lib/64" "/opt/MVS/lib/64" "/opt/MVS/lib")

if(NOT MVS_INCLUDE_DIR)
  message(FATAL_ERROR
    "找不到 MvCameraControl.h。\n"
    "  - 确认 MVS 装在 /opt/MVS，否则用 -DMVS_ROOT=/你的路径\n"
    "  - 或手动: -DMVS_INCLUDE_DIR=/opt/MVS/include")
endif()

if(NOT MVS_LIBRARY)
  message(FATAL_ERROR
    "找不到 libMvCameraControl.so。\n"
    "  - 确认 /opt/MVS/lib/64/libMvCameraControl.so 存在\n"
    "  - 或手动: -DMVS_LIBRARY=/opt/MVS/lib/64/libMvCameraControl.so")
endif()

message(STATUS "OpenCV         : ${OpenCV_VERSION}")
message(STATUS "OpenCV include : ${OpenCV_INCLUDE_DIRS}")
message(STATUS "MVS include    : ${MVS_INCLUDE_DIR}")
message(STATUS "MVS library    : ${MVS_LIBRARY}")

# ---------------------------------------------------------------------------
#  可执行文件（名字保持 YoloCamera，和你 Qt Creator 里的构建目标对得上）
# ---------------------------------------------------------------------------
add_executable(YoloCamera ${MAIN_SRC})

target_include_directories(YoloCamera PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${MVS_INCLUDE_DIR})

target_link_libraries(YoloCamera PRIVATE
    ${OpenCV_LIBS}
    ${MVS_LIBRARY}
    Threads::Threads)

# 把 /opt/MVS/lib/64 写进 RPATH，省得每次 export LD_LIBRARY_PATH
set_target_properties(YoloCamera PROPERTIES
    BUILD_RPATH   "/opt/MVS/lib/64"
    INSTALL_RPATH "/opt/MVS/lib/64")
