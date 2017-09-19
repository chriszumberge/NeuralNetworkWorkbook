using NeuralNetwork.ImageRecognition;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ImageRecognition.CodeProject
{
    class Program
    {
        static void Main(string[] args)
        {
            string imgPath = @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\img\PATTERNS";
            //string imgPath = @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\img\SYMBOLS";
            //string imgPath = @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\img\ICONS";

            int NumOfPatterns, av_ImageHeight, av_ImageWidth, inputUnit, hiddenUnit;

            InitializeSettings(imgPath, out NumOfPatterns, out av_ImageHeight, out av_ImageWidth, out inputUnit, out hiddenUnit);

            Dictionary<string, double[]> TrainingSet = GenerateTrainingSet(imgPath, av_ImageHeight, av_ImageWidth);

            NeuralNetwork<string> neuralNetwork = CreateNeuralNetwork(NumOfPatterns, av_ImageHeight, av_ImageWidth, inputUnit, hiddenUnit, TrainingSet);

            bool loaded = false;
            Console.WriteLine("Load Pre-Trained Network? (Y/N)");
            if (Console.ReadLine().ToUpper().Equals("Y"))
            {
                Console.Write("File name: ");
                string fileName = Console.ReadLine();
                neuralNetwork.LoadNetwork($@"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\Networks\{fileName}.net");
                loaded = true;
            }
            else
            {
                bool isSuccess = TrainNeuralNetwork(neuralNetwork);
            }

            string testFilePath;

            testFilePath = @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\img\PATTERNS\A.bmp";
            TestRecognition(testFilePath, av_ImageHeight, av_ImageWidth, neuralNetwork);

            testFilePath = @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\img\PATTERNS\Q.bmp";
            TestRecognition(testFilePath, av_ImageHeight, av_ImageWidth, neuralNetwork);

            testFilePath = @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\img\PATTERNS\8.bmp";
            TestRecognition(testFilePath, av_ImageHeight, av_ImageWidth, neuralNetwork);

            //testFilePath = @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\img\ICONS\Mozilla.bmp";
            //TestRecognition(testFilePath, av_ImageHeight, av_ImageWidth, neuralNetwork);

            Console.ReadLine();

            if (!loaded)
            {
                Console.WriteLine("Save Network? (Y/N)");

                if (Console.ReadLine().ToUpper().Equals("Y"))
                {
                    Console.Write("File name: ");
                    string fileName = Console.ReadLine();
                    neuralNetwork.SaveNetwork($@"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.CodeProject\Networks\{fileName}.net");
                }
            }
        }

        private static void TestRecognition(string testFilePath, int av_ImageHeight, int av_ImageWidth, NeuralNetwork<string> neuralNetwork)
        {
            string MatchedHigh = " ? ", MatchedLow = "?";
            double OutputValueHigh = 0, OutputValueLow = 0;

            Bitmap img = new Bitmap(testFilePath);

            ShowRecognitionResult(av_ImageHeight, av_ImageWidth, neuralNetwork, ref MatchedHigh, ref MatchedLow, ref OutputValueHigh, ref OutputValueLow, img);
        }

        private static bool TrainNeuralNetwork(NeuralNetwork<string> neuralNetwork)
        {
            Console.WriteLine("Began Training Process..");

            neuralNetwork.IterationChanged += NeuralNetwork_IterationChanged;

            bool isSuccess = neuralNetwork.Train();

            if (isSuccess)
                Console.WriteLine("Completed Training Process Successfully");
            else
            {
                Console.WriteLine("Training Process is Aborted or Exceed Maximum Iteration\r\n");
                Console.ReadLine();
            }

            return isSuccess;
        }

        private static NeuralNetwork<string> CreateNeuralNetwork(int NumOfPatterns, int av_ImageHeight, int av_ImageWidth, int inputUnit, int hiddenUnit, Dictionary<string, double[]> TrainingSet)
        {
            Console.WriteLine("Creating Neural Network..");

            int InputNum = inputUnit;
            int HiddenNum = hiddenUnit;

            //neuralNetwork.IterationChanged +=
            //    new NeuralNetwork<string>.IterationChangedCallBack(neuralNetwork_IterationChanged);

            NeuralNetwork<string> neuralNetwork = new NeuralNetwork<string>
                (new BP3Layer<string>(av_ImageHeight * av_ImageWidth, InputNum, HiddenNum, NumOfPatterns), TrainingSet);

            neuralNetwork.MaximumError = 1.0;

            Console.WriteLine("Done!");
            return neuralNetwork;
        }

        private static Dictionary<string, double[]> GenerateTrainingSet(string imgPath, int av_ImageHeight, int av_ImageWidth)
        {
            Console.WriteLine("Generating Training Set..");

            string[] Patterns = Directory.GetFiles(imgPath, "*.bmp");

            Dictionary<string, double[]> TrainingSet = new Dictionary<string, double[]>(Patterns.Length);

            foreach (string s in Patterns)
            {
                Bitmap Temp = new Bitmap(s);
                TrainingSet.Add(Path.GetFileNameWithoutExtension(s), ImageProcessing.ToMatrix(Temp, av_ImageHeight, av_ImageWidth));
                Temp.Dispose();
            }

            Console.WriteLine("Done!");
            return TrainingSet;
        }

        private static void InitializeSettings(string imgPath, out int NumOfPatterns, out int av_ImageHeight, out int av_ImageWidth, out int inputUnit, out int hiddenUnit)
        {
            Console.WriteLine("Initializing Settings..");

            string[] Images = Directory.GetFiles(imgPath, "*.bmp");
            NumOfPatterns = Images.Length;
            av_ImageHeight = 0;
            av_ImageWidth = 0;
            foreach (string s in Images)
            {
                Bitmap Temp = new Bitmap(s);
                av_ImageHeight += Temp.Height;
                av_ImageWidth += Temp.Width;
                Temp.Dispose();
            }
            av_ImageHeight /= NumOfPatterns;
            av_ImageWidth /= NumOfPatterns;

            int networkInput = av_ImageHeight * av_ImageWidth;

            inputUnit = (int)((networkInput + NumOfPatterns) * 0.33);
            hiddenUnit = (int)((networkInput + NumOfPatterns) * 0.11);
            Console.WriteLine("Done!");
        }

        private static void ShowRecognitionResult(int av_ImageHeight, int av_ImageWidth, NeuralNetwork<string> neuralNetwork, ref string MatchedHigh, ref string MatchedLow, ref double OutputValueHigh, ref double OutputValueLow, Bitmap img)
        {
            double[] input = ImageProcessing.ToMatrix(img, av_ImageHeight, av_ImageWidth);

            neuralNetwork.Recognize(input, ref MatchedHigh, ref OutputValueHigh, ref MatchedLow, ref OutputValueLow);

            string txtMatchedHigh = "High: " + MatchedHigh + " (%" + ((int)100 * OutputValueHigh).ToString("##") + ")";
            string txtMatchedLow = "Low: " + MatchedLow + " (%" + ((int)100 * OutputValueLow).ToString("##") + ")";

            Console.WriteLine(txtMatchedHigh);
            Console.WriteLine(txtMatchedLow);
        }

        private static void NeuralNetwork_IterationChanged(object o, NeuralEventArgs args)
        {
            Console.WriteLine($"Iteration: {args.CurrentIteration}, Error: {args.CurrentError}");
        }
    }
}
