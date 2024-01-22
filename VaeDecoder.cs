using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using Noexia.MyAI.Onnx;

namespace StableDiffusion.ML.OnnxRuntime
{
    public static class VaeDecoder
    {
        public static Tensor<float> Decoder(
            List<NamedOnnxValue> input, string vaeDecoderOnnxPath,
            EExecutionProvider executionProvider, int deviceIndex,
            out InferenceSession vaeDecodeSession, out IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output)
        {
            System.Diagnostics.Debug.WriteLine("Create VAEDecoder session...");
            var sessionOptions = SessionOptionsFactory.CreateSession(executionProvider, deviceIndex);

            // Create an InferenceSession from the Model Path.
            vaeDecodeSession = new InferenceSession(vaeDecoderOnnxPath, sessionOptions);

           // Run session and send the input data in to get inference output. 
            output = vaeDecodeSession.Run(input);
            var result = (output.ElementAt(0).Value as Tensor<float>);

            sessionOptions.Dispose();

            return result;
        }

        // create method to convert float array to an image with imagesharp
        public static Image<Rgba32> ConvertToImage(Tensor<float> output, string imageOutputPath, int width = 512, int height = 512)
        {
            var result = new Image<Rgba32>(width, height);

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }

            var imageName = $"sd_image_{DateTime.Now.ToString("yyyyMMddHHmmss")}.png";
            var imagePath = Path.Combine(Directory.GetCurrentDirectory(), imageOutputPath, imageName);

            result.Save(imagePath);

            System.Diagnostics.Debug.WriteLine($"Image saved to: {imagePath}");

            return result;
        }
    }
}
