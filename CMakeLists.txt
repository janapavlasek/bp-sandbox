cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bp_sandbox)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SIMPLE_WS_DIR "/usr/local/include/")

find_package(Eigen3 REQUIRED)
find_package(Boost 1.54.0 COMPONENTS system thread coroutine context REQUIRED)
find_package(OpenSSL REQUIRED)

include_directories(
  ${SIMPLE_WS_DIR}/simple-websocket-server
  ${EIGEN3_INCLUDE_DIR}
  ${OPENSSL_INCLUDE_DIR}
)

add_executable(bp_websocket src/server.cpp
  ${SIMPLE_WS_DIR}/simple-websocket-server/client_ws.hpp
  ${SIMPLE_WS_DIR}/simple-websocket-server/server_ws.hpp
)
target_link_libraries(bp_websocket
  ${Boost_LIBRARIES}
  ${OPENSSL_LIBRARIES}
)

if (CMAKE_BUILD_TYPE MATCHES Test)
endif()
