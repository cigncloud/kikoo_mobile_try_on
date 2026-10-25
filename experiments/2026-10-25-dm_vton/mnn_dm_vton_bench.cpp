// Two resident DM-VTON stages with synchronized outputs and refreshed inputs.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/Backend.hpp"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using Values = std::vector<float>;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}
Values read(const std::string& path, size_t count) {
    Values values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    if (!file || file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Input size mismatch: " + path);
    return values;
}
void save(const std::string& path, const Values& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!file) throw std::runtime_error("Cannot write " + path);
}
class Model {
    std::unique_ptr<MNN::Interpreter> net;
    MNN::Session* session;
    bool first = true;
public:
    double setup;
    float memory = 0;
    int selected;
    std::map<int, int> backends;
    Model(const std::string& path, int backend, int precision, const std::string& cache) {
        auto begin = Clock::now();
        net.reset(MNN::Interpreter::createFromFile(path.c_str()));
        if (!net) throw std::runtime_error("Cannot load " + path);
        if (!cache.empty()) net->setCacheFile(cache.c_str());
        net->setSessionMode(MNN::Interpreter::Session_Debug);
        net->setSessionHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
        MNN::BackendConfig options;
        options.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
        options.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = static_cast<MNNForwardType>(backend);
        config.numThread = backend == MNN_FORWARD_OPENCL ? 80 : 4;
        config.backendConfig = &options;
        session = net->createSession(config);
        if (!session) throw std::runtime_error("Cannot create session: " + path);
        int actual[16]; std::fill(actual, actual + 16, -1);
        net->getSessionInfo(session, MNN::Interpreter::BACKENDS, actual);
        selected = actual[0];
        if (selected != backend && !(backend == 0 && selected == MNN_FORWARD_CPU_EXTENSION))
            throw std::runtime_error("Requested backend unavailable");
        setup = elapsed(begin);
    }
    void put(const char* name, const Values& values) {
        auto input = net->getSessionInput(session, name);
        if (!input || input->elementSize() != static_cast<int>(values.size()))
            throw std::runtime_error(std::string("Input shape mismatch: ") + name);
        MNN::Tensor host(input, MNN::Tensor::CAFFE);
        std::copy(values.begin(), values.end(), host.host<float>());
        if (!input->copyFromHostTensor(&host)) throw std::runtime_error("Upload failed");
    }
    std::map<std::string, Values> run() {
        MNN::ErrorCode status;
        if (first) {
            auto before = [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; };
            auto after = [&](const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo*) {
                if (!outputs.empty()) {
                    auto backend = net->getBackend(session, outputs[0]);
                    backends[backend ? static_cast<int>(backend->type()) : -1]++;
                }
                return true;
            };
            status = net->runSessionWithCallBackInfo(session, before, after, true);
        } else status = net->runSession(session);
        if (status != MNN::NO_ERROR) throw std::runtime_error("Inference failed");
        std::map<std::string, Values> results;
        for (const auto& item : net->getSessionOutputAll(session)) {
            MNN::Tensor host(item.second, MNN::Tensor::CAFFE);
            if (!item.second->copyToHostTensor(&host)) throw std::runtime_error("Download failed");
            auto& values = results[item.first];
            values.assign(host.host<float>(), host.host<float>() + host.elementSize());
            for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite output");
        }
        net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);
        if (first) { net->releaseModel(); first = false; }
        return results;
    }
    void saveCache() { net->updateCacheFile(session); }
};
struct Timing { double warp, generator, total; };
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        if (argc != 7 && argc != 8) throw std::runtime_error("Usage: mnn_dm_vton_bench model_dir data_dir output_dir backend precision runs [cache_dir]");
        auto coldBegin = Clock::now();
        const std::string models = argv[1], data = argv[2], out = argv[3];
        const int backend = std::stoi(argv[4]), precision = std::stoi(argv[5]), runs = std::stoi(argv[6]);
        const std::string cache = argc == 8 ? argv[7] : "";
        auto cachePath = [&](const std::string& name) {
            return cache.empty() ? "" : cache + "/" + name + "_p" + std::to_string(precision) + ".cache";
        };
        if (runs < 1) throw std::runtime_error("runs must be positive");
        auto person = read(data + "/dmvton_input_person.f32", 3*256*192);
        auto clothes = read(data + "/dmvton_input_clothes.f32", 3*256*192);
        auto edge = read(data + "/dmvton_input_edge.f32", 256*192);
        Model warp(models + "/dmvton_warp.mnn", backend, precision, cachePath("warp"));
        Model generator(models + "/dmvton_generator.mnn", backend, precision, cachePath("generator"));
        std::vector<Timing> timings;
        Values firstResult, lastResult;
        double cold = 0;
        for (int i=0; i<=runs; ++i) {
            auto begin = Clock::now();
            warp.put("person", person); warp.put("clothes", clothes); warp.put("edge", edge);
            auto warped = warp.run();
            const double warpSeconds = elapsed(begin);
            auto genBegin = Clock::now();
            generator.put("person", person);
            generator.put("warped_cloth", warped.at("warped_cloth"));
            generator.put("warped_edge", warped.at("warped_edge"));
            auto generated = generator.run();
            const double genSeconds = elapsed(genBegin), total = elapsed(begin);
            lastResult = std::move(generated.at("tryon"));
            if (lastResult.size() != 3*256*192) throw std::runtime_error("Unexpected image shape");
            if (i == 0) { cold = elapsed(coldBegin); firstResult = lastResult; }
            timings.push_back({warpSeconds, genSeconds, total});
            std::cout << "run=" << i << " total_seconds=" << total << "\n";
        }
        save(out + "/first_tryon.f32", firstResult);
        save(out + "/tryon.f32", lastResult);
        if (!cache.empty()) { warp.saveCache(); generator.saveCache(); }
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> w(buffer);
        w.StartObject();
        w.Key("status"); w.String("completed");
        w.Key("resolution"); w.StartArray(); w.Int(256); w.Int(192); w.EndArray();
        w.Key("backend_requested"); w.Int(backend);
        w.Key("precision"); w.Int(precision);
        w.Key("kernel_cache_directory"); w.String(cache.c_str());
        w.Key("cold_to_first_output_seconds"); w.Double(cold);
        w.Key("process_peak_rss_bytes"); w.Uint64(static_cast<uint64_t>(usage.ru_maxrss)*1024);
        w.Key("input_refresh_each_forward"); w.Bool(true);
        w.Key("release_model_after_first_forward"); w.Bool(true);
        w.Key("models"); w.StartArray();
        for (auto model : {&warp, &generator}) {
            w.StartObject();
            w.Key("backend_selected"); w.Int(model->selected);
            w.Key("setup_seconds"); w.Double(model->setup);
            w.Key("mnn_session_memory_mb"); w.Double(model->memory);
            w.Key("first_forward_output_backend_counts"); w.StartObject();
            for (const auto& item : model->backends) { w.Key(std::to_string(item.first).c_str()); w.Int(item.second); }
            w.EndObject(); w.EndObject();
        }
        w.EndArray();
        w.Key("runs"); w.StartArray();
        for (size_t i=0; i<timings.size(); ++i) {
            w.StartObject();
            w.Key("warmup"); w.Bool(i==0);
            w.Key("warp_seconds"); w.Double(timings[i].warp);
            w.Key("generator_seconds"); w.Double(timings[i].generator);
            w.Key("total_seconds"); w.Double(timings[i].total);
            w.EndObject();
        }
        w.EndArray();
        w.Key("scope"); w.String("Resident warp+generator inference, input uploads, synchronized stage downloads and host bridge included. JPEG preprocessing, PNG encoding and file/ADB transfer excluded. RSS is not dedicated GPU VRAM. Tensor backend audit does not prove kernel-only GPU execution.");
        w.EndObject();
        std::ofstream metrics(out + "/metrics.json"); metrics << buffer.GetString() << '\n';
        if (!metrics) throw std::runtime_error("Cannot save metrics");
        std::cout << buffer.GetString() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "ERROR: " << error.what() << '\n'; return 1; }
}
