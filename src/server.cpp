#include <memory>
#include <future>
#include <random>

#include <simple-websocket-server/client_ws.hpp>
#include <simple-websocket-server/server_ws.hpp>
#define PI 3.141592

using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;
using WsClient = SimpleWeb::SocketClient<SimpleWeb::WS>;

int main() {
  // WebSocket (WS)-server at port 8080 using 1 thread
  WsServer server;
  server.config.port = 8080;

  // Example 1: echo WebSocket endpoint
  // Added debug messages for example use of the callbacks
  // Test with the following JavaScript:
  //   var ws=new WebSocket("ws://localhost:8080/echo");
  //   ws.onmessage=function(evt){console.log(evt.data);};
  //   ws.send("test");
  auto &bp_socket = server.endpoint["^/bp/?$"];

  bp_socket.on_message = [](std::shared_ptr<WsServer::Connection> connection, std::shared_ptr<WsServer::InMessage> in_message) {
    auto string_msg = in_message->string();

    std::cout << "Server: Message received: \"" << string_msg << "\" from " << connection.get() << std::endl;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<int> int_dist(0, 340);
    std::uniform_real_distribution<float> dist(-180, 180);

    std::string out_msg = "{\"circles\": [";
    for (size_t i = 0; i < 10; ++i)
    {
        out_msg += "[" + std::to_string(int_dist(gen)) + "," + std::to_string(int_dist(gen)) + "],";
    }
    out_msg.pop_back();
    out_msg += "],";

    for (size_t i = 0; i < 8; ++i)
    {
        out_msg += "\"l" + std::to_string(i + 1) + "\": [";
        for (size_t i = 0; i < 10; ++i)
        {
            out_msg += "[" + std::to_string(int_dist(gen)) + "," + std::to_string(int_dist(gen)) + "," + std::to_string(dist(gen)) + "],";
        }
        out_msg.pop_back();
        out_msg += "],";
    }

    out_msg.pop_back();
    out_msg += "}";

    std::cout << "Server: Sending message \"" << out_msg << "\" to " << connection.get() << std::endl;

    // connection->send is an asynchronous function
    connection->send(out_msg, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        std::cout << "Server: Error sending message. " <<
            // See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html, Error Codes for error code meanings
            "Error: " << ec << ", error message: " << ec.message() << std::endl;
      }
    });

  };

  bp_socket.on_open = [](std::shared_ptr<WsServer::Connection> connection) {
    std::cout << "Server: Opened connection " << connection.get() << std::endl;
  };

  // See RFC 6455 7.4.1. for status codes
  bp_socket.on_close = [](std::shared_ptr<WsServer::Connection> connection, int status, const std::string & /*reason*/) {
    std::cout << "Server: Closed connection " << connection.get() << " with status code " << status << std::endl;
  };

  // Can modify handshake response headers here if needed
  bp_socket.on_handshake = [](std::shared_ptr<WsServer::Connection> /*connection*/, SimpleWeb::CaseInsensitiveMultimap & /*response_header*/) {
    return SimpleWeb::StatusCode::information_switching_protocols; // Upgrade to websocket
  };

  // See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html, Error Codes for error code meanings
  bp_socket.on_error = [](std::shared_ptr<WsServer::Connection> connection, const SimpleWeb::error_code &ec) {
    std::cout << "Server: Error in connection " << connection.get() << ". "
         << "Error: " << ec << ", error message: " << ec.message() << std::endl;
  };

  // Start server and receive assigned port when server is listening for requests
  std::promise<unsigned short> server_port;
  std::thread server_thread([&server, &server_port]() {
    // Start server
    server.start([&server_port](unsigned short port) {
      server_port.set_value(port);
    });
  });
  std::cout << "Server listening on port " << server_port.get_future().get() << std::endl << std::endl;

  server_thread.join();
}
