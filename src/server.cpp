#include <memory>
#include <future>

#include <simple-websocket-server/client_ws.hpp>
#include <simple-websocket-server/server_ws.hpp>

#include "inference/particle_filter.h"

#include "server_utils.h"

using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;
using WsClient = SimpleWeb::SocketClient<SimpleWeb::WS>;

class ServerHelper
{
public:
    ServerHelper()
    {
    }

    BPSandbox::ParticleFilter pf;

    void sendParticleMessage(std::shared_ptr<WsServer::Connection>& connection, const ParticleMessage& msg)
    {
        connection->send(msg.toJSONString(), [](const SimpleWeb::error_code &ec) {
        if(ec) {
            std::cout << "Server: Error sending message. " <<
                // See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html, Error Codes for error code meanings
                "Error: " << ec << ", error message: " << ec.message() << std::endl;
            }
        });
    }

    void handleServerMessage(std::shared_ptr<WsServer::Connection>& connection, InMessageHelper& in_msg)
    {
        if (in_msg.hasKey("action"))
        {
            if (in_msg.getVal("action") == "init")
            {
                std::cout << "Server: Sending initialize message to " << connection.get() << std::endl;
                // connection->send is an asynchronous function
                int num_particles = 10;
                if (in_msg.hasKey("num_particles")) num_particles = std::stoi(in_msg.getVal("num_particles"));
                bool use_obs = true;
                if (in_msg.hasKey("init_informed")) use_obs = std::stoi(in_msg.getVal("init_informed")) == 1;

                ParticleMessage msg;
                msg.setParticles(pf.init(num_particles, use_obs));

                sendParticleMessage(connection, msg);
            }
            else if (in_msg.getVal("action") == "update")
            {
                std::cout << "Running one update" << std::endl;

                ParticleMessage msg;
                msg.setParticles(pf.update());
                sendParticleMessage(connection, msg);

                std::cout << "Done" << std::endl;
            }
            else if (in_msg.getVal("action") == "estimate")
            {
                std::cout << "Running one update" << std::endl;

                ParticleMessage msg;
                msg.setParticles(pf.estimate());
                sendParticleMessage(connection, msg);

                std::cout << "Done" << std::endl;
            }
            else
            {
                std::cout << "Action " << in_msg.getVal("action") << "is unknown." << std::endl;
            }
        }
        else
        {
            std::cout << "Nothing to do." << std::endl;
        }
    }
};


int main() {
  // WebSocket (WS)-server at port 8080 using 1 thread
  WsServer server;
  server.config.port = 8080;
  std::shared_ptr<ServerHelper> helper = std::make_shared<ServerHelper>();

  // Init web socket.
  auto &bp_socket = server.endpoint["^/bp/?$"];

  bp_socket.on_message = [&, helper](std::shared_ptr<WsServer::Connection> connection, std::shared_ptr<WsServer::InMessage> in_message) {
    auto string_msg = in_message->string();

    std::cout << "Server: Message received: \"" << string_msg << "\" from " << connection.get() << std::endl;
    InMessageHelper in_msg(string_msg);

    helper->handleServerMessage(connection, in_msg);
  };

  // Setup some basic functions.
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
