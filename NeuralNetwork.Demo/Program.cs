//
// Program.cs
//
//
// Author:
//       Chris Zumberge <chriszumberge@gmail.com>
//
// Copyright (c) 2017 Christopher Zumberge
//
// All rights reserved
//
using System;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Examples;
using NeuralNetwork.Training;

namespace NeuralNetwork.Demo
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            RunSimpleNeuralNetwork();
            //RunLayeredNeuralNetwork();
            //RunOONeuralNetwork();
            //RunOONeuralNetworkTrainingDiagnostics();
            //PlayRockPaperScizzors();
        }

        private static void PlayRockPaperScizzors()
        {
            NeuralNetwork nn = new NeuralNetwork(new Random(), 6, 1, 8, 4);

            // Rock , Paper, Scizzors
            //double[,] trainin_set_input = new double[6, 6];
            double[,] training_set_input_array = new double[,]
            {
                {1, 0, 0, 0, 1, 0},
                {1, 0, 0, 0, 0, 1},
                {0, 1, 0, 1, 0, 0},
                {0, 1, 0, 0, 0, 1},
                {0, 0, 1, 1, 0, 0},
                {0, 0, 1, 0, 1, 0}
            };

            Matrix<double> training_set_inputs = Matrix<double>.Build.DenseOfArray(training_set_input_array);
            double[,] training_set_output_array = new double[,]
            {
                { 0 }, { 1 }, { 1 }, { 0 }, { 0 }, { 1 }
            };
            Matrix<double> training_set_outputs = Matrix<double>.Build.DenseOfArray(training_set_output_array);

            nn.Train(training_set_inputs, training_set_outputs, 100000);

            List<Matrix<double>> outputs = new List<Matrix<double>>();
            // Rock vs Paper- should be 0
            Matrix<double> newInput = Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0, 0, 0, 1, 0 } });
            nn.Think(newInput, out outputs);
            Console.WriteLine(outputs.Last());
            // Rock vs Scizzors- should be 1
            newInput = Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0, 0, 0, 0, 1 } });
            nn.Think(newInput, out outputs);
            Console.WriteLine(outputs.Last());

            Console.ReadLine();
        }

        private static void RunOONeuralNetwork()
        {
            NeuralNetwork nn = new NeuralNetwork(new Random(), 3, 1, 4, 2);

            double[,] training_set_input_array = new double[,]
            {
                {0, 0, 1 },
                {0, 1, 1 },
                {1, 0, 1 },
                {0, 1, 0 },
                {1, 0, 0 },
                {1, 1, 1 },
                {0, 0, 0 }
            };
            Matrix<double> training_set_inputs = Matrix<double>.Build.DenseOfArray(training_set_input_array);
            double[,] training_set_output_array = new double[,]
            {
                { 0 }, { 1 }, { 1 }, { 1 }, { 1 }, { 0 }, { 0 }
            };
            Matrix<double> training_set_outputs = Matrix<double>.Build.DenseOfArray(training_set_output_array);

            nn.Train(training_set_inputs, training_set_outputs, 100000);

            List<Matrix<double>> outputs = new List<Matrix<double>>();
            Matrix<double> newInput = Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1, 0 } });


            nn.Think(newInput, out outputs);

            Console.WriteLine(outputs.Last());

            Console.ReadLine();
        }

        private static void RunOONeuralNetworkTrainingDiagnostics()
        {
            // -1 just selects best, otherwise give number
            int maxAvgError = -1;

            double[,] training_set_input_array = new double[,]
            {
                {0, 0, 1 },
                {0, 1, 1 },
                {1, 0, 1 },
                {0, 1, 0 },
                {1, 0, 0 },
                {1, 1, 1 },
                {0, 0, 0 }
            };
            Matrix<double> training_set_inputs = Matrix<double>.Build.DenseOfArray(training_set_input_array);
            double[,] training_set_output_array = new double[,]
            {
                { 0 }, { 1 }, { 1 }, { 1 }, { 1 }, { 0 }, { 0 }
            };
            Matrix<double> training_set_outputs = Matrix<double>.Build.DenseOfArray(training_set_output_array);

            Random random = new Random();
            NeuralNetwork nn1 = new NeuralNetwork(random, 3, 1);
            NeuralNetwork nn2 = new NeuralNetwork(random, 3, 1, 1);
            NeuralNetwork nn3 = new NeuralNetwork(random, 3, 1, 2);
            NeuralNetwork nn4 = new NeuralNetwork(random, 3, 1, 3);
            NeuralNetwork nn5 = new NeuralNetwork(random, 3, 1, 4, 1);
            NeuralNetwork nn6 = new NeuralNetwork(random, 3, 1, 4, 2);
            NeuralNetwork nn7 = new NeuralNetwork(random, 3, 1, 4, 3);

            int numTrainingIterations = 100000;

            List<TrainingSet> trainingSets = new List<TrainingSet>();
            trainingSets.Add(new TrainingSet(nn1, training_set_inputs, training_set_outputs, numTrainingIterations));
            trainingSets.Add(new TrainingSet(nn2, training_set_inputs, training_set_outputs, numTrainingIterations));
            trainingSets.Add(new TrainingSet(nn3, training_set_inputs, training_set_outputs, numTrainingIterations));
            trainingSets.Add(new TrainingSet(nn4, training_set_inputs, training_set_outputs, numTrainingIterations));
            trainingSets.Add(new TrainingSet(nn5, training_set_inputs, training_set_outputs, numTrainingIterations));
            trainingSets.Add(new TrainingSet(nn6, training_set_inputs, training_set_outputs, numTrainingIterations));
            trainingSets.Add(new TrainingSet(nn7, training_set_inputs, training_set_outputs, numTrainingIterations));


            for (int i = 0; i < trainingSets.Count; i++)
            {
                Console.WriteLine("Training Set #" + (i + 1).ToString());
                Console.WriteLine(String.Join(", ", trainingSets[i].Network.GetNumberOfNeuronsPerLayer()));
                Console.WriteLine(String.Concat("+/-", trainingSets[i].AverageError));
                Console.WriteLine(String.Concat(trainingSets[i].AverageProcessingTime.TotalMilliseconds, " ms per op"));
                Console.WriteLine(String.Concat(trainingSets[i].TrainingTime.TotalMilliseconds, " ms to train"));
                Console.WriteLine();
            }

            List<TrainingSet> variationSets = new List<TrainingSet>();

            TrainingSet topPerformingSet = null;
            topPerformingSet = trainingSets.Where(x => x.AverageError < maxAvgError).OrderBy(x => x.AverageProcessingTime).FirstOrDefault();
            // If this is null, that means there were none with an error under the max error so just grab the first one we can find
            if (topPerformingSet == null)
            {
                topPerformingSet = trainingSets.OrderBy(x => x.AverageError).First();
            }

            variationSets.Add(topPerformingSet);

            List<NeuronLayer> topPerformingHiddenLayers = topPerformingSet.Network.Layers.GetRange(1, topPerformingSet.Network.Layers.Count - 2);

            if (topPerformingHiddenLayers.Count > 0)
            {
                List<int> variation1 = topPerformingHiddenLayers.Select(x => x.NumberOfNeurons).ToList();
                if (variation1.Count > 0)
                {
                    variation1[0] += 1;
                }
                NeuralNetwork topVar1 = new NeuralNetwork(random, 3, 1, variation1.ToArray());
                variationSets.Add(new TrainingSet(topVar1, training_set_inputs, training_set_outputs, numTrainingIterations));

                List<int> variation2 = topPerformingHiddenLayers.Select(x => x.NumberOfNeurons).ToList();
                if (variation2.Count > 0 && variation2[0] > 0)
                {
                    variation2[0] -= 1;
                }
                NeuralNetwork topVar2 = new NeuralNetwork(random, 3, 1, variation2.ToArray());
                variationSets.Add(new TrainingSet(topVar2, training_set_inputs, training_set_outputs, numTrainingIterations));
            }

            int avgDifference = (int)Math.Round((topPerformingSet.Network.mNumInputs + topPerformingSet.Network.mNumOuputs) / 2.0);
            for (int i = avgDifference - 2; i <= avgDifference + 2; i++)
            {
                if (i > 0)
                {
                    List<int> variation = topPerformingHiddenLayers.Select(x => x.NumberOfNeurons).ToList();
                    variation.Add(i);
                    NeuralNetwork varNN = new NeuralNetwork(random, 3, 1, variation.ToArray());
                    variationSets.Add(new TrainingSet(varNN, training_set_inputs, training_set_outputs, numTrainingIterations));
                }
            }

            for (int i = 0; i < variationSets.Count; i++)
            {
                Console.WriteLine(String.Join(", ", variationSets[i].Network.GetNumberOfNeuronsPerLayer()));
                Console.WriteLine(String.Concat("+/-", variationSets[i].AverageError));
                Console.WriteLine(String.Concat(variationSets[i].AverageProcessingTime.TotalMilliseconds, " ms per op"));
                Console.WriteLine(String.Concat(variationSets[i].TrainingTime.TotalMilliseconds, " ms to train"));
                Console.WriteLine();
            }

            TrainingSet finalTopSet = variationSets.OrderBy(x => x.AverageError).First();

            Console.WriteLine("Winner");
            Console.WriteLine(String.Join(", ", finalTopSet.Network.GetNumberOfNeuronsPerLayer()));
            Console.WriteLine(String.Concat("+/-", finalTopSet.AverageError));
            Console.WriteLine(String.Concat(finalTopSet.AverageProcessingTime.TotalMilliseconds, " ms per op"));
            Console.WriteLine(String.Concat(finalTopSet.TrainingTime.TotalMilliseconds, " ms to train"));
            Console.WriteLine();


            //nn.Train(training_set_inputs, training_set_outputs, 100000);

            //List<Matrix<double>> outputs = new List<Matrix<double>>();
            //Matrix<double> newInput = Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1, 0 } });


            //nn.Think(newInput, out outputs);

            //Console.WriteLine(outputs.Last());

            Console.ReadLine();
        }

        private static void RunLayeredNeuralNetwork()
        {
            // 4 neurons with 3 inputs
            var layer1 = new LayeredNeuralNetwork.NeuronLayer(6, 3);
            // 5 neurons with 4 inputs
            //var layer2 = new LayeredNeuralNetwork.NeuronLayer(1, 4);
            //var layer2 = new LayeredNeuralNetwork.NeuronLayer(2, 4);
            var layer2 = new LayeredNeuralNetwork.NeuronLayer(5, 6);
            var layer3 = new LayeredNeuralNetwork.NeuronLayer(4, 5);
            var layer4 = new LayeredNeuralNetwork.NeuronLayer(3, 4);
            var layer5 = new LayeredNeuralNetwork.NeuronLayer(2, 3);
            var layer6 = new LayeredNeuralNetwork.NeuronLayer(1, 2);

            //LayeredNeuralNetwork nn = new LayeredNeuralNetwork(layer1, layer2);
            LayeredNeuralNetwork nn = new LayeredNeuralNetwork(layer1, layer2, layer3, layer4, layer5, layer6);

            Console.WriteLine("Random starting synaptic weights: ");

            double[,] training_set_input_array = new double[,]
            {
                {0, 0, 1 },
                {0, 1, 1 },
                {1, 0, 1 },
                {0, 1, 0 },
                {1, 0, 0 },
                {1, 1, 1 },
                {0, 0, 0 }
            };
            Matrix<double> training_set_inputs = Matrix<double>.Build.DenseOfArray(training_set_input_array);
            //List<Vector<double>> training_set_inputs = new List<Vector<double>>
            //{
            //    Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1}),
            //    Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 1}),
            //    Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 1}),
            //    Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0}),
            //    Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0}),
            //    Vector<double>.Build.DenseOfArray(new double[] { 1, 1, 1}),
            //    Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0})
            //};

            //double[] training_set_output_array = new double[]
            //{
            //    0, 1, 1, 1, 1, 0, 0
            //};
            //Vector<double> training_set_outputs = Vector<double>.Build.DenseOfArray(training_set_output_array);
            double[,] training_set_output_array = new double[,]
            {
                { 0 }, { 1 }, { 1 }, { 1 }, { 1 }, { 0 }, { 0 }
            };
            //double[,] training_set_output_array = new double[,]
            //{
            //    { 0, 1}, { 1, 1 }, { 1, 1 }, { 1, 0 }, { 1, 0 }, { 0, 1 }, { 0, 0 }
            //};
            Matrix<double> training_set_outputs = Matrix<double>.Build.DenseOfArray(training_set_output_array);

            nn.Train(training_set_inputs, training_set_outputs, 100000);

            //Console.WriteLine("New Synaptic Weights");
            //Console.WriteLine("Layer 1 (4 neurons, each with 3 inputs)");
            //Console.WriteLine(nn.Layer1.SynapticWeights.ToString());
            //Console.WriteLine("Layer 2 (1 neuron with 4 inputs)");
            //Console.WriteLine(nn.Layer2.SynapticWeights.ToString());

            //Matrix<double> hidden_state = Matrix<double>.Build.Dense(1, 1, 0);
            //Matrix<double> output = Matrix<double>.Build.Dense(1, 1, 0);
            List<Matrix<double>> outputs = new List<Matrix<double>>();

            Console.WriteLine("Training Set 1 -> 0");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0, 1 } }), out outputs);
            Console.WriteLine(outputs.Last());

            Console.WriteLine("Training Set 2 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 1 } }), out outputs);
            Console.WriteLine(outputs.Last());

            Console.WriteLine("Training Set 3 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0, 1 } }), out outputs);
            Console.WriteLine(outputs.Last());

            Console.WriteLine("Training Set 4 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 0 } }), out outputs);
            Console.WriteLine(outputs.Last());

            Console.WriteLine("Training Set 5 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0, 0 } }), out outputs);
            Console.WriteLine(outputs.Last());

            Console.WriteLine("Training Set 6 -> 0");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1, 1 } }), out outputs);
            Console.WriteLine(outputs.Last());

            Console.WriteLine("Training Set 7 -> 0");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0, 0 } }), out outputs);
            Console.WriteLine(outputs.Last());

            Console.WriteLine("Considering new situation [1, 1, 0]");

            Matrix<double> newInput = Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1, 0 } });


            nn.Think(newInput, out outputs);

            Console.WriteLine(outputs.Last());

            Console.ReadLine();
        }

        private static void RunSimpleNeuralNetwork()
        {
            SimpleNeuralNetwork nn = new SimpleNeuralNetwork(3);

            Console.WriteLine("Random starting synaptic weights: ");
            Console.WriteLine(nn.SynapticWeights.ToString());

            // Training set, 4 exmaples of 3 inputs and 1 output
            double[,] training_set_input_array = new double[,]
            {
                {0, 0, 1 },
                {1, 1, 1 },
                {1, 0, 1 },
                {0, 1, 1 }
            };
            Matrix<double> training_set_inputs = Matrix<double>.Build.DenseOfArray(training_set_input_array);

            double[] training_set_output_array = new double[]
            {
                0, 1, 1, 0
            };
            Vector<double> training_set_outputs = Vector<double>.Build.DenseOfArray(training_set_output_array);

            // Train the neural network using a training set.
            // Do it 10,000 times making small adjustmetns each time
            nn.Train(training_set_inputs, training_set_outputs, 1000000);

            Console.WriteLine("New synaptic weights after training: ");
            Console.WriteLine(nn.SynapticWeights.ToString());

            Console.WriteLine("Training Set 1 -> 0");
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1 })));

            Console.WriteLine("Training Set 2 -> 1");
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 1, 1, 1 })));

            Console.WriteLine("Training Set 3 -> 1");
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 1 })));

            Console.WriteLine("Training Set 4 -> 0");
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 1 })));


            // Test with new situation
            Console.WriteLine("Considering new situation [1, 0, 0]");
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0 })));

            Console.WriteLine("Considering new situation [1, 1, 0]");
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 1, 1, 0 })));

            Console.WriteLine("Considering new situation [0, 1, 0]");
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0 })));

            Console.ReadLine();
        }

        public static string VectorToString(double[] vector)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");

            foreach (double val in vector)
            {
                sb.AppendLine(String.Concat("[", val, "]"));
            }

            sb.Append("]");
            return sb.ToString();
        }
    }
}
