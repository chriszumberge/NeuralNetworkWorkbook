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

namespace NeuralNetwork.Demo
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            SimpleNeuralNetwork nn = new SimpleNeuralNetwork(3);

            Console.WriteLine("Random starting synaptic weights: ");
            Console.WriteLine(nn.SynapticWeights.ToString());

            // Training set, 4 exmaples of 3 inputs and 1 output
            //double[][] training_set_inputs = new double[][]
            //{
            //    new double[] { 0, 0, 1 },
            //    new double[] { 1, 1, 1 },
            //    new double[] { 1, 0, 1 },
            //    new double[] { 0, 1, 1 }
            //};
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
            nn.Train(training_set_inputs, training_set_outputs, 100000);

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
            //Console.WriteLine(nn.Think(new double[] { 1, 0, 0 }));
            Console.WriteLine(nn.Think(Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0 })));

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
