using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Noexia.MyAI.Onnx
{
    public static class SessionOptionsFactory
    {
        public static SessionOptions CreateSession(EExecutionProvider executionProvider, int deviceId = 0)
        {
            Console.WriteLine("Create session");
            System.Diagnostics.Debug.WriteLine("Create session");
            var sessionOptions = new SessionOptions();


            switch (executionProvider)
            {
                case EExecutionProvider.DirectML:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.EnableMemoryPattern = false;
                    sessionOptions.AppendExecutionProvider_DML(deviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case EExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                default:
                case EExecutionProvider.Cuda:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    //default to CUDA, fall back on CPU if CUDA is not available.
                    sessionOptions.AppendExecutionProvider_CUDA(deviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    //sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                    return sessionOptions;

            }

        }
    }
}
