// Run real model inputs on Android and retain output tensors and measurements.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/Backend.hpp"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
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
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 2) { std::cerr << "Usage: mnn_component_bench manifest.json\n"; return 2; }
    try {
        std::ifstream stream(argv[1]);
        std::string text((std::istreambuf_iterator<char>(stream)), {});
        rapidjson::Document config;
        config.Parse(text.c_str());
        if (config.HasParseError()) throw std::runtime_error("Invalid manifest");
        int requested = config["backend"].GetInt();
        int runs = config["runs"].GetInt();
        if (runs < 1) throw std::runtime_error("runs must be positive");
        auto start = Clock::now();
        std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(config["model"].GetString()));
        if (!net) throw std::runtime_error("Model loading failed");
        double loadSeconds = seconds(start);
        net->setSessionMode(MNN::Interpreter::Session_Debug);
        net->setSessionHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
        MNN::BackendConfig backend;
        backend.precision = config.HasMember("precision") ?
            static_cast<MNN::BackendConfig::PrecisionMode>(config["precision"].GetInt()) : MNN::BackendConfig::Precision_Low;
        backend.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig schedule;
        schedule.type = static_cast<MNNForwardType>(requested);
        schedule.numThread = requested == MNN_FORWARD_OPENCL ?
            (config.HasMember("gpu_mode") ? config["gpu_mode"].GetInt() : (MNN_GPU_TUNING_FAST | MNN_GPU_MEMORY_IMAGE)) : 4;
        schedule.backendConfig = &backend;
        start = Clock::now();
        auto session = net->createSession(schedule);
        if (!session) throw std::runtime_error("Session creation failed");
        double sessionSeconds = seconds(start);
        int actualBackends[16];
        std::fill(actualBackends, actualBackends + 16, -1);
        net->getSessionInfo(session, MNN::Interpreter::BACKENDS, actualBackends);
        if (actualBackends[0] != requested && !(requested == MNN_FORWARD_CPU && actualBackends[0] == MNN_FORWARD_CPU_EXTENSION)) throw std::runtime_error("Requested backend was not selected");
        start = Clock::now();
        std::map<std::string, std::unique_ptr<MNN::Tensor>> hostInputs;
        for (auto& item : config["inputs"].GetArray()) {
            auto input = net->getSessionInput(session, item["name"].GetString());
            if (!input) throw std::runtime_error("Missing input tensor");
            auto& host = hostInputs[item["name"].GetString()];
            host.reset(new MNN::Tensor(input, MNN::Tensor::CAFFE));
            std::ifstream file(item["file"].GetString(), std::ios::binary);
            file.read(reinterpret_cast<char*>(host->host<float>()), host->size());
            if (!file || file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Input size mismatch");
            if (!input->copyFromHostTensor(host.get())) throw std::runtime_error("Input upload failed");
        }
        double inputSeconds = seconds(start);
        auto uploadInputs = [&]() {
            for (auto& item : hostInputs) {
                auto input = net->getSessionInput(session, item.first.c_str());
                if (!input->copyFromHostTensor(item.second.get())) throw std::runtime_error("Input refresh failed");
            }
        };
        std::map<std::string, int> opBackends;
        auto before = [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; };
        auto after = [&](const std::vector<MNN::Tensor*>& outputs, const MNN::OperatorInfo* info) {
            if (!outputs.empty()) {
                auto b = net->getBackend(session, outputs[0]);
                opBackends[info->name()] = b ? static_cast<int>(b->type()) : -1;
            }
            return true;
        };
        std::map<std::string, std::unique_ptr<MNN::Tensor>> hostOutputs;
        for (auto& output : net->getSessionOutputAll(session)) {
            hostOutputs[output.first].reset(new MNN::Tensor(output.second, MNN::Tensor::CAFFE));
        }
        start = Clock::now();
        auto status = net->runSessionWithCallBackInfo(session, before, after, true);
        if (status != MNN::NO_ERROR) throw std::runtime_error("Warmup inference failed");
        for (auto& output : net->getSessionOutputAll(session)) {
            if (!output.second->copyToHostTensor(hostOutputs[output.first].get())) throw std::runtime_error("Warmup output download failed");
        }
        double warmupSeconds = seconds(start);
        std::vector<double> times;
        std::vector<double> refreshTimes;
        for (int i = 0; i < runs; ++i) {
            start = Clock::now();
            uploadInputs();
            refreshTimes.push_back(seconds(start));
            start = Clock::now();
            if (net->runSession(session) != MNN::NO_ERROR) throw std::runtime_error("Inference failed");
            for (auto& output : net->getSessionOutputAll(session)) {
                if (!output.second->copyToHostTensor(hostOutputs[output.first].get())) throw std::runtime_error("Output download failed");
            }
            times.push_back(seconds(start));
            std::cout << "run " << i + 1 << ": " << times.back() << " seconds\n" << std::flush;
        }
        size_t nonfinite = 0;
        for (auto& item : config["outputs"].GetArray()) {
            auto& host = hostOutputs.at(item["name"].GetString());
            for (int i = 0; i < host->elementSize(); ++i) nonfinite += !std::isfinite(host->host<float>()[i]);
            std::ofstream file(item["file"].GetString(), std::ios::binary);
            file.write(reinterpret_cast<const char*>(host->host<float>()), host->size());
            if (!file) throw std::runtime_error("Output save failed");
        }
        float memory = 0;
        net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);
        rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> w(buffer);
        w.StartObject();
        w.Key("status"); w.String(nonfinite ? "nonfinite_output" : "completed");
        w.Key("pid"); w.Int(getpid());
        w.Key("backend_requested"); w.Int(requested);
        w.Key("backend_selected"); w.Int(actualBackends[0]);
        w.Key("gpu_mode"); w.Int(schedule.numThread);
        w.Key("precision_mode"); w.Int(backend.precision);
        w.Key("warmup_output_synchronized"); w.Bool(true);
        w.Key("inputs_refreshed_each_run"); w.Bool(true);
        w.Key("model_load_seconds"); w.Double(loadSeconds);
        w.Key("session_create_seconds"); w.Double(sessionSeconds);
        w.Key("input_upload_seconds"); w.Double(inputSeconds);
        w.Key("warmup_seconds"); w.Double(warmupSeconds);
        w.Key("inference_and_output_download_seconds"); w.StartArray();
        for (auto t : times) w.Double(t);
        w.EndArray();
        w.Key("input_refresh_seconds"); w.StartArray();
        for (auto t : refreshTimes) w.Double(t);
        w.EndArray();
        w.Key("mnn_session_memory_mb"); w.Double(memory);
        w.Key("process_peak_rss_bytes"); w.Uint64(static_cast<uint64_t>(usage.ru_maxrss) * 1024);
        w.Key("output_nonfinite_elements"); w.Uint64(nonfinite);
        w.Key("operator_output_backends"); w.StartObject();
        for (auto& op : opBackends) { w.Key(op.first.c_str()); w.Int(op.second); }
        w.EndObject();
        w.Key("memory_note"); w.String("RSS includes GPU mappings visible in /proc/self/status, but not all possible driver allocations. MNN allocator size can omit OpenCL pool/driver allocations. This is one component, not full generation.");
        w.EndObject();
        std::ofstream result(config["metrics"].GetString());
        result << buffer.GetString() << '\n';
        std::cout << buffer.GetString() << '\n';
        return nonfinite ? 3 : 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
