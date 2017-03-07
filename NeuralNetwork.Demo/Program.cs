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

namespace NeuralNetwork.Demo
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            //RunSimpleNeuralNetwork();
            RunLayeredNeuralNetwork();
        }

        private static void RunLayeredNeuralNetwork()
        {
            // 4 neurons with 3 inputs
            var layer1 = new LayeredNeuralNetwork.NeuronLayer(4, 3);
            // 1 neuron with 4 inputs
            var layer2 = new LayeredNeuralNetwork.NeuronLayer(1, 4);

            LayeredNeuralNetwork nn = new LayeredNeuralNetwork(layer1, layer2);

            Console.WriteLine("Random starting synaptic weights: ");
            //Console.WriteLine(nn.SynapticWeights.ToString());
            Console.WriteLine("Layer 1 (4 neurons, each with 3 inputs)");
            Console.WriteLine(nn.Layer1.SynapticWeights.ToString());
            Console.WriteLine("Layer 2 (1 neuron with 4 inputs)");
            Console.WriteLine(nn.Layer2.SynapticWeights.ToString());


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
            Matrix<double> training_set_outputs = Matrix<double>.Build.DenseOfArray(training_set_output_array);

            nn.Train(training_set_inputs, training_set_outputs, 100000);

            Console.WriteLine("New Synaptic Weights");
            Console.WriteLine("Layer 1 (4 neurons, each with 3 inputs)");
            Console.WriteLine(nn.Layer1.SynapticWeights.ToString());
            Console.WriteLine("Layer 2 (1 neuron with 4 inputs)");
            Console.WriteLine(nn.Layer2.SynapticWeights.ToString());

            Matrix<double> hidden_state = Matrix<double>.Build.Dense(1, 1, 0);
            Matrix<double> output = Matrix<double>.Build.Dense(1, 1, 0);

            Console.WriteLine("Training Set 1 -> 0");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0, 1 } }), out hidden_state, out output);
            Console.WriteLine(output);

            Console.WriteLine("Training Set 2 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 1 } }), out hidden_state, out output);
            Console.WriteLine(output);

            Console.WriteLine("Training Set 3 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0, 1 } }), out hidden_state, out output);
            Console.WriteLine(output);

            Console.WriteLine("Training Set 4 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 0 } }), out hidden_state, out output);
            Console.WriteLine(output);

            Console.WriteLine("Training Set 5 -> 1");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0, 0 } }), out hidden_state, out output);
            Console.WriteLine(output);

            Console.WriteLine("Training Set 6 -> 0");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1, 1 } }), out hidden_state, out output);
            Console.WriteLine(output);

            Console.WriteLine("Training Set 7 -> 0");
            nn.Think(Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0, 0 } }), out hidden_state, out output);
            Console.WriteLine(output);

            Console.WriteLine("Considering new situation [1, 1, 0]");

            Matrix<double> newInput = Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 1, 0 } });
           

            nn.Think(newInput, out hidden_state, out output);

            Console.WriteLine(output);

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
