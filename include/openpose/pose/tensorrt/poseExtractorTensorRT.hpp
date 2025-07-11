#ifndef OPENPOSE_POSE_POSE_EXTRACTOR_TENSOR_RT_HPP
#define OPENPOSE_POSE_POSE_EXTRACTOR_TENSOR_RT_HPP

#include <openpose/core/common.hpp>
#include <openpose/net/bodyPartConnectorCaffe.hpp> // Reuse existing body part connector
#include <openpose/net/maximumCaffe.hpp>           // Reuse existing maximum implementation
#include <openpose/net/tensorrt/netTensorRT.hpp>   // Our new TensorRT network
#include <openpose/net/nmsCaffe.hpp>               // Reuse existing NMS implementation
#include <openpose/net/resizeAndMergeCaffe.hpp>    // Reuse existing resize/merge
#include <openpose/pose/enumClasses.hpp>
#include <openpose/pose/poseExtractorNet.hpp>

namespace op
{
    /**
     * TensorRT implementation of pose extraction for OpenPose
     * This class provides the same interface as PoseExtractorCaffe but uses TensorRT for inference
     */
    class OP_API PoseExtractorTensorRT : public PoseExtractorNet
    {
    public:
        /**
         * Constructor for TensorRT pose extractor
         * @param poseModel The pose model type (BODY_25, COCO_18, MPI_15, etc.)
         * @param engineFolder Folder containing TensorRT engine files
         * @param gpuId GPU device ID to use for inference
         * @param heatMapTypes Types of heat maps to generate
         * @param heatMapScaleMode Scale mode for heat maps
         * @param addPartCandidates Whether to add part candidates
         * @param maximizePositives Whether to maximize positives
         * @param enginePath Optional specific path to engine file (overrides automatic path)
         * @param precision Precision mode used for the engine ("fp32", "fp16", "int8")
         * @param enableNet Whether to enable the network (for debugging)
         * @param enableProfiling Whether to enable TensorRT profiling
         */
        PoseExtractorTensorRT(
            const PoseModel poseModel,
            const std::string &engineFolder,
            const int gpuId,
            const std::vector<HeatMapType> &heatMapTypes = {},
            const ScaleMode heatMapScaleMode = ScaleMode::ZeroToOneFixedAspect,
            const bool addPartCandidates = false,
            const bool maximizePositives = false,
            const std::string &enginePath = "",
            const std::string &precision = "fp16",
            const bool enableNet = true,
            const bool enableProfiling = false);

        virtual ~PoseExtractorTensorRT();

        /**
         * Initialize the TensorRT engine on the current thread
         * Must be called before any inference operations
         */
        virtual void netInitializationOnThread() override;

        /**
         * Perform forward pass inference using TensorRT
         * @param inputNetData Input tensor data for multiple scales
         * @param inputDataSize Original input image size
         * @param scaleInputToNetInputs Scale factors from input to network inputs
         * @param poseNetOutput Optional pre-computed pose network output
         */
        virtual void forwardPass(
            const std::vector<Array<float>> &inputNetData,
            const Point<int> &inputDataSize,
            const std::vector<double> &scaleInputToNetInputs = {1.f},
            const Array<float> &poseNetOutput = Array<float>{}) override;

        // Accessor methods (inherited from PoseExtractorNet)
        virtual const float *getCandidatesCpuConstPtr() const override;
        virtual const float *getCandidatesGpuConstPtr() const override;
        virtual const float *getHeatMapCpuConstPtr() const override;
        virtual const float *getHeatMapGpuConstPtr() const override;
        virtual std::vector<int> getHeatMapSize() const override;
        virtual const float *getPoseGpuConstPtr() const override;

        /**
         * Get TensorRT engine information for debugging
         * @return String containing engine details
         */
        std::string getEngineInfo() const;

        /**
         * Get performance profiling results
         * @return String containing profiling information
         */
        std::string getProfilingResults() const;

        /**
         * Enable or disable TensorRT profiling
         * @param enable True to enable profiling, false to disable
         */
        void setProfilingEnabled(bool enable);

    private:
        // Configuration parameters
        const PoseModel mPoseModel;
        const int mGpuId;
        const std::string mEngineFolder;
        const std::string mEnginePathOverride;
        const std::string mPrecision;
        const bool mEnableNet;
        const bool mEnableProfiling;

        // TensorRT network components
        std::vector<std::shared_ptr<Net>> spNets;
        std::shared_ptr<ResizeAndMergeCaffe<float>> spResizeAndMergeCaffe;
        std::shared_ptr<NmsCaffe<float>> spNmsCaffe;
        std::shared_ptr<BodyPartConnectorCaffe<float>> spBodyPartConnectorCaffe;
        std::shared_ptr<MaximumCaffe<float>> spMaximumCaffe;

        // Network state
        std::vector<std::vector<int>> mNetInput4DSizes;
        std::vector<std::shared_ptr<ArrayCpuGpu<float>>> spTensorRTNetOutputBlobs;
        std::shared_ptr<ArrayCpuGpu<float>> spHeatMapsBlob;
        std::shared_ptr<ArrayCpuGpu<float>> spPeaksBlob;
        std::shared_ptr<ArrayCpuGpu<float>> spMaximumPeaksBlob;

        /**
         * Get the engine file path for the given pose model
         * @param poseModel The pose model type
         * @param engineFolder Base folder containing engine files
         * @param precision Precision mode ("fp32", "fp16", "int8")
         * @return Full path to the engine file
         */
        std::string getEngineFilePath(const PoseModel poseModel,
                                      const std::string &engineFolder,
                                      const std::string &precision) const;

        /**
         * Add TensorRT network to the processing pipeline
         * @param net Vector of networks to add to
         * @param tensorrtNetOutputBlob Vector of output blobs to add to
         * @param poseModel Pose model type
         * @param gpuId GPU device ID
         * @param engineFolder Engine folder path
         * @param enginePath Optional specific engine path
         * @param precision Precision mode
         * @param enableProfiling Whether to enable profiling
         */
        void addTensorRTNetOnThread(
            std::vector<std::shared_ptr<Net>> &net,
            std::vector<std::shared_ptr<ArrayCpuGpu<float>>> &tensorrtNetOutputBlob,
            const PoseModel poseModel,
            const int gpuId,
            const std::string &engineFolder,
            const std::string &enginePath,
            const std::string &precision,
            const bool enableProfiling);

        // Disable copy constructor and assignment
        DELETE_COPY(PoseExtractorTensorRT);
    };
}

#endif // OPENPOSE_POSE_POSE_EXTRACTOR_TENSOR_RT_HPP