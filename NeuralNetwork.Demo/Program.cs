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

namespace NeuralNetwork.Demo
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            SimpleNeuralNetwork nn = new SimpleNeuralNetwork(3);

            Console.WriteLine("Random starting synaptic weights: ");
            Console.WriteLine(VectorToString(nn.SynapticWeights));

            // Training set, 4 exmaples of 3 inputs and 1 output
            double[][] training_set_inputs = new double[][]
            {
                new double[] { 0, 0, 1 },
                new double[] { 1, 1, 1 },
                new double[] { 1, 0, 1 },
                new double[] { 0, 1, 1 }
            };
            //double[,] training_set_inputs = new double[,]
            //{
            //    {0, 0, 1 },
            //    {1, 1, 1 },
            //    {1, 0, 1 },
            //    {0, 1, 1 }
            //};

            double[] training_set_outputs = new double[]
            {
                0, 1, 1, 0
            };

            // Train the neural network using a training set.
            // Do it 10,000 times making small adjustmetns each time
            nn.Train(training_set_inputs, training_set_outputs, 1000000);

            Console.WriteLine("New synaptic weights after training: ");
            Console.WriteLine(VectorToString(nn.SynapticWeights));

            // Test with new situation
            Console.WriteLine("Considering new situation [1, 0, 0]");
            Console.WriteLine(nn.Think(new double[] { 1, 0, 0 }));

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
