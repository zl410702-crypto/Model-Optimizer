// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace
{
class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

template <typename T>
struct TrtDestroy
{
    void operator()(T* value) const noexcept
    {
        if (value != nullptr)
        {
            value->destroy();
        }
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDestroy<T>>;

bool save(std::string const& path, void const* data, std::size_t size)
{
    std::ofstream output(path, std::ios::binary);
    output.write(static_cast<char const*>(data), static_cast<std::streamsize>(size));
    return output.good();
}

void printUsage(char const* program)
{
    std::cerr << "Usage: " << program
              << " model.onnx output.plan layer_info.json [workspace_mib]\n";
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 5)
    {
        printUsage(argv[0]);
        return 2;
    }

    std::string const onnxPath{argv[1]};
    std::string const enginePath{argv[2]};
    std::string const layerInfoPath{argv[3]};
    std::uint64_t workspaceMiB = 1024;
    if (argc == 5)
    {
        try
        {
            workspaceMiB = std::stoull(argv[4]);
        }
        catch (std::exception const& error)
        {
            std::cerr << "Invalid workspace size: " << error.what() << '\n';
            return 2;
        }
    }

    Logger logger;
    TrtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    std::uint32_t const networkFlags = 1U << static_cast<std::uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtPtr<nvinfer1::INetworkDefinition> network{
        builder == nullptr ? nullptr : builder->createNetworkV2(networkFlags)};
    TrtPtr<nvonnxparser::IParser> parser{
        network == nullptr ? nullptr : nvonnxparser::createParser(*network, logger)};
    if (builder == nullptr || network == nullptr || parser == nullptr)
    {
        std::cerr << "Failed to create TensorRT builder, network, or ONNX parser.\n";
        return 1;
    }

    if (!parser->parseFromFile(
            onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ONNX parsing failed with " << parser->getNbErrors() << " errors:\n";
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cerr << "  " << parser->getError(i)->desc() << '\n';
        }
        return 1;
    }

    TrtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (config == nullptr)
    {
        std::cerr << "Failed to create TensorRT builder config.\n";
        return 1;
    }
    config->setMaxWorkspaceSize(static_cast<std::size_t>(workspaceMiB) << 20);
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

    std::cout << "Building INT8 engine from Q/DQ ONNX model.\n"
              << "Workspace limit: " << workspaceMiB << " MiB\n";
    TrtPtr<nvinfer1::IHostMemory> serialized{
        builder->buildSerializedNetwork(*network, *config)};
    if (serialized == nullptr)
    {
        std::cerr << "TensorRT engine build failed.\n";
        return 1;
    }
    if (!save(enginePath, serialized->data(), serialized->size()))
    {
        std::cerr << "Failed to save engine to " << enginePath << '\n';
        return 1;
    }

    TrtPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    TrtPtr<nvinfer1::ICudaEngine> engine{
        runtime == nullptr ? nullptr
                           : runtime->deserializeCudaEngine(serialized->data(), serialized->size())};
    std::unique_ptr<nvinfer1::IEngineInspector> inspector{
        engine == nullptr ? nullptr : engine->createEngineInspector()};
    if (engine == nullptr || inspector == nullptr)
    {
        std::cerr << "Engine was built, but TensorRT could not create its inspector.\n";
        return 1;
    }

    char const* const json =
        inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    if (json == nullptr || !save(layerInfoPath, json, std::char_traits<char>::length(json)))
    {
        std::cerr << "Failed to export layer information to " << layerInfoPath << '\n';
        return 1;
    }

    std::cout << "Engine: " << enginePath << " (" << serialized->size() << " bytes)\n"
              << "Layer info: " << layerInfoPath << '\n';
    return 0;
}
