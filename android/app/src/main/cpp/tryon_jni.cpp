#include <jni.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {
using Values = std::vector<float>;
using Clock = std::chrono::steady_clock;
constexpr int Pixels = 192 * 256;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string stringValue(JNIEnv* env, jstring value) {
    const char* bytes = env->GetStringUTFChars(value, nullptr);
    if (!bytes) throw std::runtime_error("Cannot read path");
    std::string result(bytes);
    env->ReleaseStringUTFChars(value, bytes);
    return result;
}
Values floats(JNIEnv* env, jfloatArray array, int size) {
    if (!array || env->GetArrayLength(array) != size) throw std::invalid_argument("Unexpected tensor size");
    Values result(size);
    env->GetFloatArrayRegion(array, 0, size, result.data());
    for (float v : result) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite input");
    return result;
}
class Stage {
    std::unique_ptr<MNN::Interpreter> net;
    MNN::Session* session = nullptr;
    bool first = true;
public:
    Stage(const std::string& model, const std::string& cache) {
        net.reset(MNN::Interpreter::createFromFile(model.c_str()));
        if (!net) throw std::runtime_error("Cannot load model");
        net->setCacheFile(cache.c_str());
        net->setSessionHint(MNN::Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
        MNN::BackendConfig options;
        options.precision = MNN::BackendConfig::Precision_High;
        options.memory = MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig config;
        config.type = MNN_FORWARD_OPENCL;
        config.numThread = MNN_GPU_TUNING_FAST | MNN_GPU_MEMORY_BUFFER;
        config.backendConfig = &options;
        session = net->createSession(config);
        if (!session) throw std::runtime_error("Cannot create GPU session");
        int selected[16]; std::fill(selected, selected + 16, -1);
        net->getSessionInfo(session, MNN::Interpreter::BACKENDS, selected);
        if (selected[0] != MNN_FORWARD_OPENCL) throw std::runtime_error("OpenCL GPU unavailable in app");
    }
    void put(const char* name, const Values& values) {
        auto input = net->getSessionInput(session, name);
        if (!input || input->elementSize() != static_cast<int>(values.size())) throw std::runtime_error("Input contract mismatch");
        MNN::Tensor host(input, MNN::Tensor::CAFFE);
        std::copy(values.begin(), values.end(), host.host<float>());
        if (!input->copyFromHostTensor(&host)) throw std::runtime_error("GPU upload failed");
    }
    std::map<std::string, Values> run() {
        if (net->runSession(session) != MNN::NO_ERROR) throw std::runtime_error("GPU inference failed");
        std::map<std::string, Values> result;
        for (const auto& output : net->getSessionOutputAll(session)) {
            MNN::Tensor host(output.second, MNN::Tensor::CAFFE);
            if (!output.second->copyToHostTensor(&host)) throw std::runtime_error("GPU download failed");
            auto& values = result[output.first];
            values.assign(host.host<float>(), host.host<float>() + host.elementSize());
            for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite output");
        }
        if (first) {
            net->updateCacheFile(session);
            net->releaseModel();
            first = false;
        }
        return result;
    }
};
class Engine {
public:
    Stage warp, generator;
    double warpSeconds = 0, generatorSeconds = 0, totalSeconds = 0;
    Engine(const std::string& models, const std::string& cache)
        : warp(models + "/dmvton_warp.mnn", cache + "/warp_fp32.cache"),
          generator(models + "/dmvton_generator.mnn", cache + "/generator_fp32.cache") {}
    Values run(const Values& person, const Values& clothes, const Values& edge) {
        const auto begin = Clock::now();
        warp.put("person", person); warp.put("clothes", clothes); warp.put("edge", edge);
        auto outputs = warp.run();
        warpSeconds = elapsed(begin);
        const auto genBegin = Clock::now();
        generator.put("person", person);
        generator.put("warped_cloth", outputs.at("warped_cloth"));
        generator.put("warped_edge", outputs.at("warped_edge"));
        auto result = generator.run().at("tryon");
        generatorSeconds = elapsed(genBegin);
        totalSeconds = elapsed(begin);
        if (result.size() != 3 * Pixels) throw std::runtime_error("Output contract mismatch");
        return result;
    }
};
Engine& engine(jlong handle) {
    if (!handle) throw std::invalid_argument("Engine is closed");
    return *reinterpret_cast<Engine*>(handle);
}
void failure(JNIEnv* env, const std::exception& error) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_mobiletryon_demo_TryOnEngine_nativeCreate(JNIEnv* env, jclass, jstring model, jstring cache) {
    try { return reinterpret_cast<jlong>(new Engine(stringValue(env, model), stringValue(env, cache))); }
    catch (const std::exception& error) { failure(env, error); return 0; }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_mobiletryon_demo_TryOnEngine_nativeRun(JNIEnv* env, jclass, jlong handle, jfloatArray person, jfloatArray clothes, jfloatArray edge) {
    try {
        auto result = engine(handle).run(floats(env, person, 3 * Pixels), floats(env, clothes, 3 * Pixels), floats(env, edge, Pixels));
        auto output = env->NewFloatArray(result.size());
        if (output) env->SetFloatArrayRegion(output, 0, result.size(), result.data());
        return output;
    } catch (const std::exception& error) { failure(env, error); return nullptr; }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobiletryon_demo_TryOnEngine_nativeStats(JNIEnv* env, jclass, jlong handle) {
    try {
        auto& e = engine(handle);
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        const double stats[] = {e.warpSeconds, e.generatorSeconds, e.totalSeconds, static_cast<double>(usage.ru_maxrss) * 1024};
        auto output = env->NewDoubleArray(4);
        if (output) env->SetDoubleArrayRegion(output, 0, 4, stats);
        return output;
    } catch (const std::exception& error) { failure(env, error); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobiletryon_demo_TryOnEngine_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<Engine*>(handle);
}
