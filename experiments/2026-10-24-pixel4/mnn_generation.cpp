// Fixed-resolution, sequential-CFG Mobile-VTON inference with cached garments.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/Backend.hpp"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
using Values = std::vector<float>;
constexpr int H = 128, W = 96, C = 16;
float halfToFloat(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
    int exponent = (half >> 10) & 31;
    uint32_t fraction = half & 1023, bits;
    if (exponent == 0 && fraction) {
        exponent = -14;
        while (!(fraction & 1024)) { fraction <<= 1; --exponent; }
        bits = sign | (static_cast<uint32_t>(exponent + 127) << 23) | ((fraction & 1023) << 13);
    } else if (exponent == 0) bits = sign;
    else if (exponent == 31) bits = sign | 0x7f800000 | (fraction << 13);
    else bits = sign | (static_cast<uint32_t>(exponent + 112) << 23) | (fraction << 13);
    float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
uint64_t peakRSS() {
    rusage usage{}; getrusage(RUSAGE_SELF, &usage);
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
}
uint64_t currentRSS() {
    std::ifstream file("/proc/self/status"); std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") == 0) return std::stoull(line.substr(6)) * 1024;
    }
    return 0;
}
Values readFloats(const std::string& path, size_t count) {
    Values result(count); std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(float));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Float file size mismatch: " + path);
    return result;
}
void finite(const Values& data) {
    for (float value : data) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite tensor");
}
void saveFloats(const std::string& path, const Values& data) {
    finite(data); std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    if (!file) throw std::runtime_error("Cannot save " + path);
}

struct RunRecord { std::string label; double seconds; uint64_t rss; };
struct ModelRecord {
    std::string path; int precision; double setup;
    std::map<int, int> backendOutputs;
};
std::vector<RunRecord> runs;
std::vector<ModelRecord> models;

class Model {
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    bool first = true;
    size_t recordIndex;
public:
    Model(const std::string& path, int precision, int requested) {
        auto begin = Clock::now();
        net.reset(MNN::Interpreter::createFromFile(path.c_str()));
        if (!net) throw std::runtime_error("Cannot load " + path);
        net->setSessionMode(MNN::Interpreter::Session_Debug);
        net->setSessionHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
        MNN::BackendConfig backend;
        backend.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig schedule;
        schedule.type = static_cast<MNNForwardType>(requested);
        schedule.numThread = requested == MNN_FORWARD_OPENCL ? 80 : 4;
        schedule.backendConfig = &backend;
        session = net->createSession(schedule);
        if (!session) throw std::runtime_error("Cannot create session: " + path);
        int actual[16]; std::fill(actual, actual + 16, -1);
        net->getSessionInfo(session, MNN::Interpreter::BACKENDS, actual);
        if (actual[0] != requested && !(requested == 0 && actual[0] == MNN_FORWARD_CPU_EXTENSION))
            throw std::runtime_error("Requested backend unavailable");
        recordIndex = models.size();
        models.push_back({path, precision, elapsed(begin), {}});
        std::cout << "session_ready " << path << " setup_seconds=" << models.back().setup << "\n";
    }
    void put(const std::string& name, const Values& values) {
        auto input = net->getSessionInput(session, name.c_str());
        if (!input || input->elementSize() != static_cast<int>(values.size()))
            throw std::runtime_error("Input shape mismatch: " + name);
        MNN::Tensor host(input, MNN::Tensor::CAFFE);
        std::copy(values.begin(), values.end(), host.host<float>());
        if (!input->copyFromHostTensor(&host)) throw std::runtime_error("Upload failed: " + name);
    }
    void putFile(const std::string& name, const std::string& path, bool half = false, bool zero = false) {
        auto input = net->getSessionInput(session, name.c_str());
        if (!input) throw std::runtime_error("Missing input: " + name);
        MNN::Tensor host(input, MNN::Tensor::CAFFE);
        const int count = host.elementSize();
        if (zero) {
            std::fill(host.host<float>(), host.host<float>() + count, 0.0f);
        } else {
            std::ifstream file(path, std::ios::binary);
            if (half) {
                std::vector<uint16_t> packed(count);
                file.read(reinterpret_cast<char*>(packed.data()), count * sizeof(uint16_t));
                for (int i = 0; i < count; ++i) host.host<float>()[i] = halfToFloat(packed[i]);
            } else {
                file.read(reinterpret_cast<char*>(host.host<float>()), count * sizeof(float));
            }
            if (!file || file.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("Input file mismatch: " + path);
        }
        if (!input->copyFromHostTensor(&host)) throw std::runtime_error("Upload failed: " + name);
    }
    Values forward(const std::string& outputName, const std::string& label) {
        auto begin = Clock::now();
        MNN::ErrorCode status;
        if (first) {
            auto before = [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; };
            auto after = [&](const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo*) {
                if (!outputs.empty()) {
                    auto backend = net->getBackend(session, outputs[0]);
                    models[recordIndex].backendOutputs[backend ? static_cast<int>(backend->type()) : -1]++;
                }
                return true;
            };
            status = net->runSessionWithCallBackInfo(session, before, after, true);
        } else status = net->runSession(session);
        if (status != MNN::NO_ERROR) throw std::runtime_error("Inference failed: " + label);
        auto output = net->getSessionOutput(session, outputName.c_str());
        if (!output) throw std::runtime_error("Missing output: " + outputName);
        MNN::Tensor host(output, MNN::Tensor::CAFFE);
        if (!output->copyToHostTensor(&host)) throw std::runtime_error("Download failed");
        Values result(host.host<float>(), host.host<float>() + host.elementSize());
        finite(result);
        if (first) {
            // Fixed shapes: no subsequent session creation/resizing is allowed.
            net->releaseModel();
            first = false;
        }
        runs.push_back({label, elapsed(begin), currentRSS()});
        std::cout << label << " seconds=" << runs.back().seconds << " rss_bytes=" << runs.back().rss << "\n";
        return result;
    }
};

struct Tile { int h, w; Values data; };
size_t at(int channel, int y, int x, int h, int w) {
    return (static_cast<size_t>(channel) * h + y) * w + x;
}

void encodePerson(const rapidjson::Document& config) {
    const auto& paths = config["models"];
    const std::string data = config["data"].GetString();
    auto image = readFloats(data + "/person.f32", 3 * 1024 * 768);
    std::vector<std::vector<Tile>> tiles(3);
    for (int y : {0, 384, 768}) {
        for (int x : {0, 384}) tiles[y / 384].push_back({std::min(512, 1024-y)/8, std::min(512, 768-x)/8, {}});
    }
    for (auto shape : std::vector<std::pair<int,int>>{{512,512},{512,384},{256,512},{256,384}}) {
        std::string key = "vae_" + std::to_string(shape.first) + "_" + std::to_string(shape.second);
        Model model(paths[key.c_str()].GetString(), 1, config["backend"].GetInt());
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 2; ++j) {
            auto& tile = tiles[i][j];
            if (tile.h * 8 != shape.first || tile.w * 8 != shape.second) continue;
            Values input(3 * shape.first * shape.second);
            for (int c = 0; c < 3; ++c) for (int y = 0; y < shape.first; ++y) {
                std::copy_n(image.data() + at(c, i*384+y, j*384, 1024, 768), shape.second,
                            input.data() + at(c, y, 0, shape.first, shape.second));
            }
            model.put("image", input);
            tile.data = model.forward("output_0", "vae_tile_"+std::to_string(i)+"_"+std::to_string(j));
            if (tile.data.size() != static_cast<size_t>(32 * tile.h * tile.w)) throw std::runtime_error("VAE output shape");
            saveFloats("moments_tile_"+std::to_string(i)+"_"+std::to_string(j)+".f32", tile.data);
        }
    }
    Values moments(32 * H * W);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 2; ++j) {
        auto& tile = tiles[i][j];
        if (i) {
            const auto& above = tiles[i-1][j];
            const int extent = std::min({16, above.h, tile.h});
            for (int c = 0; c < 32; ++c) for (int y = 0; y < extent; ++y) for (int x = 0; x < tile.w; ++x) {
                const float alpha = static_cast<float>(y) / extent;
                auto index = at(c,y,x,tile.h,tile.w);
                tile.data[index] = above.data[at(c,above.h-extent+y,x,above.h,above.w)]*(1-alpha) + tile.data[index]*alpha;
            }
        }
        if (j) {
            const auto& left = tiles[i][j-1];
            const int extent = std::min({16, left.w, tile.w});
            for (int c = 0; c < 32; ++c) for (int x = 0; x < extent; ++x) for (int y = 0; y < tile.h; ++y) {
                const float alpha = static_cast<float>(x) / extent;
                auto index = at(c,y,x,tile.h,tile.w);
                tile.data[index] = left.data[at(c,y,left.w-extent+x,left.h,left.w)]*(1-alpha) + tile.data[index]*alpha;
            }
        }
        for (int c = 0; c < 32; ++c) for (int y = 0; y < std::min(48,tile.h); ++y)
            std::copy_n(tile.data.data()+at(c,y,0,tile.h,tile.w), std::min(48,tile.w), moments.data()+at(c,i*48+y,j*48,H,W));
    }
    saveFloats("person_moments.f32", moments);
    auto epsilon = readFloats(data+"/person_epsilon.f32", C*H*W);
    Values person(C*H*W);
    for (size_t i = 0; i < person.size(); ++i) {
        const float logvar = std::clamp(moments[person.size()+i], -30.0f, 20.0f);
        const float sampled = moments[i] + std::exp(0.5f*logvar)*epsilon[i];
        person[i] = (sampled-0.0609f)*1.5305f;
    }
    saveFloats("person_latent.f32", person);
}

void denoise(const rapidjson::Document& config) {
    const std::string data = config["data"].GetString();
    auto person = readFloats("person_latent.f32", C*H*W);
    auto garment = readFloats(data+"/garment_latent.f32", C*H*W);
    auto noise = readFloats(data+"/initial_noise.f32", C*H*W);
    Values latents(C*2*H*W), condition(latents.size());
    for (int c = 0; c < C; ++c) {
        for (int half = 0; half < 2; ++half) std::copy_n(noise.data()+c*H*W,H*W,latents.data()+(c*2+half)*H*W);
        std::copy_n(person.data()+c*H*W,H*W,condition.data()+c*2*H*W);
        std::copy_n(garment.data()+c*H*W,H*W,condition.data()+(c*2+1)*H*W);
    }
    const float scale = config["guidance_scale"].GetFloat();
    Model model(config["models"]["tryon"].GetString(), config["tryon_precision"].GetInt(), config["backend"].GetInt());
    const int steps = config["steps"].GetInt();
    for (int step = 0; step < steps; ++step) {
        const auto begin = Clock::now();
        Values sample = latents;
        sample.insert(sample.end(), condition.begin(), condition.end());
        Values predictions[2];
        for (int branch = 0; branch < 2; ++branch) {
            const std::string name = branch == 0 ? "unconditional" : "conditional";
            model.put("sample", sample);
            model.put("timestep", {config["timesteps"][step].GetFloat()});
            model.putFile("text", data+"/text_"+name+".f32");
            model.putFile("image_embeds", data+"/image_"+name+".f32");
            char folder[40]; std::snprintf(folder,sizeof(folder),"/features_%02d/",step);
            for (int feature = 0; feature < config["feature_count"].GetInt(); ++feature) {
                const std::string key = "feature_"+std::to_string(feature);
                model.putFile(key,data+folder+key+".f16",true,branch==0);
            }
            predictions[branch] = model.forward("output_0", "step_"+std::to_string(step+1)+"_"+name);
            if (predictions[branch].size()!=latents.size()) throw std::runtime_error("Denoiser output shape");
            if (step==0) saveFloats("first_"+name+".f32",predictions[branch]);
        }
        const float delta = config["sigmas"][step+1].GetFloat()-config["sigmas"][step].GetFloat();
        for (size_t i=0;i<latents.size();++i) {
            const float difference = predictions[1][i]-predictions[0][i];
            const float weighted = scale*difference;
            const float guided = predictions[0][i]+weighted;
            const float change = delta*guided;
            latents[i] = latents[i]+change;
        }
        finite(latents);
        char path[64]; std::snprintf(path,sizeof(path),"latents_step_%02d.f32",step+1);
        saveFloats(path,latents);
        std::cout << "step_complete " << step+1 << "/" << steps << " wall_seconds=" << elapsed(begin) << "\n";
    }
    saveFloats("final_latents.f32",latents);
    Values decode(C*H*W);
    for (int c=0;c<C;++c) for (int i=0;i<H*W;++i) decode[c*H*W+i]=latents[c*2*H*W+i]/1.5305f+0.0609f;
    saveFloats("decoder_latents.f32",decode);
}

void decode(const rapidjson::Document& config) {
    Model model(config["models"]["decoder"].GetString(),2,config["backend"].GetInt());
    model.put("latent",readFloats("decoder_latents.f32",C*H*W));
    auto image=model.forward("image","decoder");
    if (image.size()!=3*1024*768) throw std::runtime_error("Decoder output shape");
    saveFloats("image.f32",image);
}

void writeMetrics(const std::string& phase, double wall) {
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> w(buffer);
    w.StartObject();
    w.Key("status"); w.String("completed");
    w.Key("phase"); w.String(phase.c_str());
    w.Key("pid"); w.Int(getpid());
    w.Key("wall_seconds"); w.Double(wall);
    w.Key("process_peak_rss_bytes"); w.Uint64(peakRSS());
    w.Key("release_model_buffer_after_first_forward"); w.Bool(true);
    w.Key("inputs_refreshed_each_forward"); w.Bool(true);
    w.Key("models"); w.StartArray();
    for (const auto& model:models) {
        w.StartObject(); w.Key("path"); w.String(model.path.c_str());
        w.Key("precision"); w.Int(model.precision);
        w.Key("setup_seconds"); w.Double(model.setup);
        w.Key("first_forward_output_backend_counts"); w.StartObject();
        for (const auto& item:model.backendOutputs) { w.Key(std::to_string(item.first).c_str()); w.Int(item.second); }
        w.EndObject(); w.EndObject();
    }
    w.EndArray(); w.Key("runs"); w.StartArray();
    for (const auto& run:runs) {
        w.StartObject(); w.Key("label"); w.String(run.label.c_str());
        w.Key("forward_download_seconds"); w.Double(run.seconds);
        w.Key("post_forward_rss_bytes"); w.Uint64(run.rss); w.EndObject();
    }
    w.EndArray();
    w.Key("scope"); w.String("On-phone neural component stages; CFG, Euler scheduler, tile blending and posterior arithmetic use CPU float32. Tensor placement audit is not kernel-level proof of GPU-only execution. Offline garment precomputation excluded.");
    w.EndObject();
    std::ofstream file(phase+"_metrics.json"); file<<buffer.GetString()<<'\n';
    if (!file) throw std::runtime_error("Cannot save metrics");
}

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        if (argc!=3) throw std::runtime_error("Usage: mnn_generation generation.json vae|denoise|decode");
        auto begin=Clock::now();
        std::ifstream file(argv[1]); std::string text((std::istreambuf_iterator<char>(file)),{});
        rapidjson::Document config; config.Parse(text.c_str());
        if (config.HasParseError() || !config.IsObject()) throw std::runtime_error("Invalid configuration");
        const std::string phase=argv[2];
        if (phase=="vae") encodePerson(config);
        else if (phase=="denoise") denoise(config);
        else if (phase=="decode") decode(config);
        else throw std::runtime_error("Unknown phase");
        writeMetrics(phase,elapsed(begin));
        return 0;
    } catch (const std::exception& error) { std::cerr<<"ERROR: "<<error.what()<<'\n'; return 1; }
}
