cmake_minimum_required(VERSION 2.8.12)

project(libNiirs)

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

#add_executable(${PROJECT_NAME}
#    "libniirs.cpp")


add_executable(${PROJECT_NAME} "cniirsmetric.cpp"
    "cniirsmetric.h"
    "main.cpp")
#    "libniirs.cpp")

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets)
