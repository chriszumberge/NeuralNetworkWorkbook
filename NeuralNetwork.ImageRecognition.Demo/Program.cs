using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Examples;
using NeuralNetwork.ImageRecognition;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ImageRecognition.Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            int averageImgHeight = 0;
            int averageImgWidth = 0;

            int totalImgHeight = 0;
            int totalImgWidth = 0;

            // Load images
            //string[] patterns = Directory.GetFiles(@"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.Demo\imgs", "*.bmp");
            //string[] patterns = Directory.GetFiles(@"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.Demo\imgs", "A.bmp");
            //string[] patterns = Directory.GetFiles(@"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.Demo\imgs\AlphaSub", "*.bmp");
            string[] patterns = Directory.GetFiles(@"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.Demo\imgs\Unified", "*.bmp");

            var trainingSetDictionary = new Dictionary<string, double[]>(patterns.Length);

            foreach (string s in patterns)
            {
                Bitmap temp = new Bitmap(s);
                totalImgHeight += temp.Height;
                totalImgWidth += temp.Width;
                temp.Dispose();
            }

            averageImgHeight = totalImgHeight / patterns.Length;
            averageImgWidth = totalImgWidth / patterns.Length;

            int numNetworkInputs = averageImgHeight * averageImgWidth;
            int numNetworkOutputs = patterns.Length;

            foreach (string s in patterns)
            {
                Bitmap temp = new Bitmap(s);
                trainingSetDictionary.Add(Path.GetFileNameWithoutExtension(s), ImageProcessing.ToMatrix(temp, averageImgHeight, averageImgWidth));
                temp.Dispose();
            }
            var trainingSetList = trainingSetDictionary.ToList();

            var layer1 = new LayeredNeuralNetwork.NeuronLayer((int)(numNetworkInputs * 1.33), numNetworkInputs);
            var layer2 = new LayeredNeuralNetwork.NeuronLayer((int)(numNetworkInputs * 0.66), (int)(numNetworkInputs * 1.33));
            var layer3 = new LayeredNeuralNetwork.NeuronLayer((int)(numNetworkInputs * 0.33), (int)(numNetworkInputs * 0.66));
            var layer4 = new LayeredNeuralNetwork.NeuronLayer((int)(numNetworkInputs * 0.1), (int)(numNetworkInputs * 0.33));
            var layer5 = new LayeredNeuralNetwork.NeuronLayer(numNetworkOutputs, (int)(numNetworkInputs * 0.1));

            LayeredNeuralNetwork nn = new LayeredNeuralNetwork(layer1, layer2, layer3, layer4, layer5);

            //List<Vector<double>> training_set_inputs = new List<Vector<double>>();
            double[,] training_set_input_array = new double[trainingSetDictionary.Count, numNetworkInputs];
            double[,] training_set_output_array = new double[numNetworkOutputs, numNetworkOutputs];

            for (int i = 0; i < trainingSetDictionary.Count; i++)
            {
                for (int j = 0; j < trainingSetList[i].Value.Length; j++)
                {
                    training_set_input_array[i, j] = trainingSetList[i].Value[j];
                }
                training_set_output_array[i, i] = 1;
            }

            Matrix<double> training_set_inputs = Matrix<double>.Build.DenseOfArray(training_set_input_array);
            Matrix<double> training_set_outputs = Matrix<double>.Build.DenseOfArray(training_set_output_array);

            nn.Train(training_set_inputs, training_set_outputs, 1000);

            List<Matrix<double>> outputs = new List<Matrix<double>>();

            Console.WriteLine("Testing A");
            double[,] valueArray = GetMatrixFromImageForTest(averageImgHeight, averageImgWidth, @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.Demo\imgs\Unified\A.bmp");
            nn.Think(Matrix<double>.Build.DenseOfArray(valueArray), out outputs);
            string guessedValue = GetGuessedValue(trainingSetList, outputs);
            Console.WriteLine(guessedValue);

            Console.WriteLine();

            Console.WriteLine("Testing B");
            valueArray = GetMatrixFromImageForTest(averageImgHeight, averageImgWidth, @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.Demo\imgs\Unified\B.bmp");
            nn.Think(Matrix<double>.Build.DenseOfArray(valueArray), out outputs);
            guessedValue = GetGuessedValue(trainingSetList, outputs);
            Console.WriteLine(guessedValue);

            Console.WriteLine();

            Console.WriteLine("Testing C");
            valueArray = GetMatrixFromImageForTest(averageImgHeight, averageImgWidth, @"C:\Users\zumberc\Documents\GitHub\NeuralNetwork\NeuralNetwork.ImageRecognition.Demo\imgs\Unified\C.bmp");
            nn.Think(Matrix<double>.Build.DenseOfArray(valueArray), out outputs);
            guessedValue = GetGuessedValue(trainingSetList, outputs);
            Console.WriteLine(guessedValue);

            Console.WriteLine();

            Console.ReadLine();
        }

        private static string GetGuessedValue(List<KeyValuePair<string, double[]>> trainingSetList, List<Matrix<double>> outputs)
        {
            var output = outputs.Last().Row(0).ToList();
            var maxIndex = output.IndexOf(output.Max());
            var guessedValue = trainingSetList.ElementAt(maxIndex).Key;
            return guessedValue;
        }

        private static double[,] GetMatrixFromImageForTest(int averageImgHeight, int averageImgWidth, string fileLocation)
        {
            Bitmap img = new Bitmap(fileLocation);
            double[] img_Val = ImageProcessing.ToMatrix(img, averageImgHeight, averageImgWidth);
            img.Dispose();
            double[,] img_Array = new double[1, img_Val.Length];
            for (int i = 0; i < img_Val.Length; i++)
            {
                img_Array[0, i] = img_Val[i];
            }

            return img_Array;
        }
    }
}
