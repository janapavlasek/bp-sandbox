#include <memory>
#include <future>
#include <random>
#include <algorithm>

#include <simple-websocket-server/client_ws.hpp>
#include <simple-websocket-server/server_ws.hpp>
#define PI 3.141592

using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;
using WsClient = SimpleWeb::SocketClient<SimpleWeb::WS>;

typedef std::vector<std::vector<float> > ParticleList;


class InMessageHelper
{
public:
    InMessageHelper(const std::string& in_msg)
    {
        parseInput(in_msg);
    }

    std::map<std::string, std::string> getData() const
    {
        return data_;
    }

    bool hasKey(const std::string k) const
    {
        return (data_.find(k) != data_.end());
    }

    std::string getVal(const std::string& k) const
    {
        std::string val = data_.at(k);
        return val;
    }

private:
    void parseInput(const std::string& in_msg)
    {
        std::cout << "Parsing incoming message..." << std::endl;
        std::string raw = in_msg;

        if (raw.find("{") == std::string::npos)
        {
            std::cout << "Incoming message is not valid: " << raw << std::endl;
            return;
        }

        // Remove first bracket.
        raw.erase(0, raw.find("{") + 1);
        while (raw.find(":") != std::string::npos)
        {
            std::string key = raw.substr(0, raw.find(":"));
            key = strip(key);
            raw.erase(0, raw.find(":") + 1);

            data_.insert({key, ""});

            std::string val;
            if (raw.find(",") != std::string::npos)
            {
                val = raw.substr(0, raw.find(","));
                val = strip(val);
                data_[key] = val;
                raw.erase(0, raw.find(",") + 1);
            }
            else
            {
                val = raw.substr(0, raw.find("}"));
                val = strip(val);
                data_[key] = val;
                break;
            }
        }
        std::cout << "Data:" << std::endl;
        for (auto const& x : data_)
        {
            std::cout << "\t" << x.first << ": " << x.second << std::endl;
        }
        std::cout << std::endl;
    }
    std::string strip(const std::string& s)
    {
        std::string r = s;
        r.erase(std::remove(r.begin(), r.end(), '\"'), r.end());
        r.erase(std::remove(r.begin(), r.end(), '\''), r.end());
        r.erase(std::remove(r.begin(), r.end(), ' '), r.end());
        return r;
    }

    std::map<std::string, std::string> data_;

};


class ParticleMessage
{
public:
    ParticleMessage() :
      algo("")
    {

    }

    std::string toJSONString() const
    {
        std::string msg = "{";
        // Algo info.
        msg += "\"algo\": \"" + algo + "\",";
        // Particles:
        for (auto const& x : particles)
        {
            msg += "\"" + x.first + "\": [";
            // Add each particle.
            for (auto& p : x.second)
            {
                msg += "[";
                for (auto& ele : p)
                {
                    msg += std::to_string(ele) + ",";
                }
                msg.pop_back();
                msg += "],";
            }
            msg.pop_back();
            msg += "],";
        }
        msg.pop_back();
        msg += "}";

        return msg;
    }

    std::string algo;
    std::map<std::string, ParticleList> particles;
};


ParticleMessage randomMessage(const int num_particles)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<int> int_dist(0, 340);
    std::uniform_real_distribution<float> dist(-180, 180);

    ParticleMessage msg;
    msg.particles.insert({"circles", ParticleList()});
    for (size_t i = 0; i < num_particles; ++i)
    {
        std::vector<float> p{int_dist(gen), int_dist(gen)};
        msg.particles["circles"].push_back(p);
    }

    for (size_t i = 0; i < 8; ++i)
    {
        std::string name = "l" + std::to_string(i + 1);
        msg.particles.insert({name, ParticleList()});
        for (size_t i = 0; i < num_particles; ++i)
        {
            std::vector<float> p{int_dist(gen), int_dist(gen), dist(gen)};
            msg.particles[name].push_back(p);
        }
    }

    return msg;
}


void sendParticleMessage(const std::shared_ptr<WsServer::Connection>& connection, const ParticleMessage& msg)
{
    connection->send(msg.toJSONString(), [](const SimpleWeb::error_code &ec) {
    if(ec) {
        std::cout << "Server: Error sending message. " <<
            // See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html, Error Codes for error code meanings
            "Error: " << ec << ", error message: " << ec.message() << std::endl;
        }
    });
}


void handleServerMessage(const std::shared_ptr<WsServer::Connection>& connection, const InMessageHelper& in_msg)
{
    if (in_msg.hasKey("action"))
    {
        if (in_msg.getVal("action") == "init")
        {
            std::cout << "Server: Sending initialize message to " << connection.get() << std::endl;
            // connection->send is an asynchronous function
            int num_particles = 10;
            if (in_msg.hasKey("num_particles")) num_particles = std::stoi(in_msg.getVal("num_particles"));
            auto msg = randomMessage(num_particles);

            sendParticleMessage(connection, msg);
        }
        else if (in_msg.getVal("action") == "start")
        {
            std::cout << "Starting..." << std::endl;
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


int main() {
  // WebSocket (WS)-server at port 8080 using 1 thread
  WsServer server;
  server.config.port = 8080;

  // Init web socket.
  auto &bp_socket = server.endpoint["^/bp/?$"];

  bp_socket.on_message = [](std::shared_ptr<WsServer::Connection> connection, std::shared_ptr<WsServer::InMessage> in_message) {
    auto string_msg = in_message->string();

    std::cout << "Server: Message received: \"" << string_msg << "\" from " << connection.get() << std::endl;
    InMessageHelper in_msg(string_msg);

    handleServerMessage(connection, in_msg);
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
