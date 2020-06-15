#ifndef BP_SANDBOX_SERVER_UTILS_H
#define BP_SANDBOX_SERVER_UTILS_H

#include <memory>
#include <future>
#include <random>
#include <algorithm>

#include <simple-websocket-server/client_ws.hpp>
#include <simple-websocket-server/server_ws.hpp>

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

    void setParticles(const std::map<std::string, ParticleList>& p)
    {
        particles = p;
    }

    std::string algo;
    std::map<std::string, ParticleList> particles;
};


ParticleMessage randomMessage(const int num_particles)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> pix_dist(0, 340);
    std::uniform_real_distribution<float> dist(-180, 180);

    ParticleMessage msg;
    msg.particles.insert({"circles", ParticleList()});
    for (size_t i = 0; i < num_particles; ++i)
    {
        std::vector<float> p{pix_dist(gen), pix_dist(gen)};
        msg.particles["circles"].push_back(p);
    }

    for (size_t i = 0; i < 8; ++i)
    {
        std::string name = "l" + std::to_string(i + 1);
        msg.particles.insert({name, ParticleList()});
        for (size_t i = 0; i < num_particles; ++i)
        {
            std::vector<float> p{pix_dist(gen), pix_dist(gen), dist(gen)};
            msg.particles[name].push_back(p);
        }
    }

    return msg;
}

#endif  // BP_SANDBOX_SERVER_UTILS_H
