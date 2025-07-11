#ifndef OPENPOSE_NET_NET_TENSOR_RT_HPP
#define OPENPOSE_NET_NET_TENSOR_RT_HPP

#include <memory>
#include <string>
#include <vector>
#include <openpose/core/common.hpp>
#include <openpose/net/net.hpp>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace op
{
    /**
     * TensorRT implementation of the Net interface for OpenPose inference
     */
    class OP_API NetTensorRT : public Net
    {
    public:
        /**
         * Constructor for TensorRT network
         * @param enginePath Path to the TensorRT engine file (.trt)
         * @param gpuId GPU device ID to use for inference
         * @param enableProfiling Enable TensorRT profiling for performance analysis
         */
        NetTensorRT(const std::string &enginePath, const int gpuId = 0,
                    const bool enableProfiling = false);

        virtual ~NetTensorRT();

        /**
         * Initialize the TensorRT engine on the current thread
         * Must be called before any inference operations
         */
        virtual void initializationOnThread() override;

        /**
         * Reshape the network input/output tensors
         * @param inputShape New input tensor shape [batch, channels, height, width]
         */
        virtual void reshapeIfNeeded(const std::vector<int> &inputShape);

        /**
         * Perform forward pass inference
         * @param inputData Input tensor data
         * @param inputDataSize Size of input data
         */
        virtual void forwardPass(const Array<float> &inputData) override;

        /**
         * Get the output tensor from the last forward pass
         * @return Shared pointer to output tensor
         */
        virtual std::shared_ptr<ArrayCpuGpu<float>> getOutputBlobArray() override;

        /**
         * Get input tensor dimensions
         * @return Vector containing input dimensions [batch, channels, height, width]
         */
        std::vector<int> getInputDimensions() const;

        /**
         * Get output tensor dimensions
         * @return Vector containing output dimensions
         */
        std::vector<int> getOutputDimensions() const;

        /**
         * Get engine information for debugging
         * @return String containing engine details
         */
        std::string getEngineInfo() const;

        /**
         * Enable/disable profiling for performance analysis
         * @param enable True to enable profiling, false to disable
         */
        void setProfilingEnabled(bool enable);

        /**
         * Get profiling results if profiling is enabled
         * @return String containing profiling information
         */
        std::string getProfilingResults() const;

    private:
        struct ImplNetTensorRT;
        std::unique_ptr<ImplNetTensorRT> upImpl;

        // Disable copy constructor and assignment
        DELETE_COPY(NetTensorRT);
    };
}

#endif // OPENPOSE_NET_NET_TENSOR_RT_HPP