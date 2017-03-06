using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class SimpleNeuralNetwork
    {
        double[] synaptic_weights { get; set; }
        public double[] SynapticWeights => synaptic_weights;

        public SimpleNeuralNetwork(int numInputs)
        {
            Random random = new Random();

            synaptic_weights = new double[numInputs];

            for (int i = 0; i < numInputs; i++)
            {
                synaptic_weights[i] = (random.NextDouble() * 2) - 1;
            }
        }

        //The Sigmoid function, normalizes between 0 and 1
        public double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        // Derivative of the Sigmoid funciton.
        // The Gradient of the Sigmoid curve.
        // It indicates how confident we are about the existing weight
        public double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }


        //public void Train(double[,] inputs, double[] expectedOutputs, int numberOfTrainingIterations)
        public void Train(double[][] inputs, double[] expectedOutputs, int numberOfTrainingIterations)
        {
            for (int t = 0; t < numberOfTrainingIterations; t++)
            {
                double[] actualOutputs = new double[inputs.Length];

                for (int i = 0; i < inputs.Length; i++)
                {
                    double[] input = inputs[i];
                    actualOutputs[i] = this.Think(input);
                }

                double[] errors = new double[expectedOutputs.Length];
                for (int i = 0; i < expectedOutputs.Length; i++)
                {
                    errors[i] = expectedOutputs[i] - actualOutputs[i];
                }

                double[] adjustedErrors = new double[expectedOutputs.Length];
                for (int i = 0; i < expectedOutputs.Length; i++)
                {
                    adjustedErrors[i] = errors[i] * SigmoidDerivative(actualOutputs[i]);
                }

                double[][] transposedInputs = Transpose(inputs);

                double[] adjustments = new double[adjustedErrors.Length];
                for (int i = 0; i < transposedInputs.Length; i++)
                {
                    double adjustment = 0;
                    double[] transposedInput = transposedInputs[i];

                    for (int j = 0; j < transposedInput.Length; j++)
                    {
                        adjustment += transposedInput[j] * adjustments[j];
                    }

                    adjustments[i] = adjustment;
                }

                for (int i = 0; i < synaptic_weights.Length; i++)
                {
                    synaptic_weights[i] += adjustments[i];
                }


                //double[,] actualOutput = Dot(inputs, ToVector(synaptic_weights));

                //double[] errors = new double[expectedOutputs.Length];
                //double[] rawAdjustments = new double[expectedOutputs.Length];
                ////double[] adjustments = new double[synaptic_weights.Length];

                //for (int i = 0; i < expectedOutputs.Length; i++)
                //{
                //    errors[i] = actualOutput[0, i] - expectedOutputs[i];
                //    rawAdjustments[i] = errors[i] * SigmoidDerivative(actualOutput[0, i]);
                //}

                //var adjustment = Dot(inputs, ToVector(rawAdjustments));

                ////double[] adjustments = Dot(inputs, rawAdjustments);

                //for (int i = 0; i < synaptic_weights.Length - 1; i++)
                //{
                //    synaptic_weights[i] += adjustment[0, i];
                //}
            }
        }

        private double[][] Transpose(double[][] array)
        {
            double[][] newArray = new double[array[0].Length][];
            for (int i = 0; i < array[0].Length; i++)
            {
                double[] newInnerArray = new double[array.Length];
                for (int j = 0; j < array.Length; j++)
                {
                    newInnerArray[j] = array[j][i];
                }
                newArray[i] = newInnerArray;
            }
            return newArray;
        }

        public double Think(double[] input)
        {
            //return Dot(input, synaptic_weights).Sum();
            double output = 0;

            for (int i = 0; i < input.Length; i++)
            {
                output += input[i] * synaptic_weights[i];
            }

            return output;
        }

        public double[] Multiply(double[] vector, double scalar)
        {
            double[] result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] * scalar;
            }
            return result;
        }

        //public double[] Dot(double[] vector, double[] multiplier)
        //{
        //    return Dot(new double[1][] { vector }, multiplier);
        //}

        //public double[] Dot(double[][] matrix, double[] multiplier)
        //{
        //    double[] result = new double[matrix.Length];

        //    for(int v = 0; v < matrix.Length; v++)
        //    {
        //        double[] vector = matrix[v];
        //        double output = 0;
        //        for (int i = 0; i < multiplier.Length; i++)
        //        {
        //            output += vector[i] * multiplier[i];
        //        }
        //        result[v] = output;
        //    }
        //    return result;
        //}

        //public double[,] Dot(double[,] matrix1, double[,] matrix2)
        //{
        //    var newMatrix = new double[matrix1.GetLength(0), matrix2.GetLength(1)];

        //    if (matrix1.GetLength(1) == matrix2.GetLength(0))
        //    {
        //        for (int i = 0; i < newMatrix.GetLength(0); i++)
        //        {
        //            for (int j = 0; j < newMatrix.GetLength(1); j++)
        //            {
        //                newMatrix[i, j] = 0;
        //                for (int k = 0; k < matrix1.GetLength(1); k++)
        //                {
        //                    newMatrix[i, j] = newMatrix[i, j] + (matrix1[i, k] * matrix2[k, j]);
        //                }
        //            }
        //        }

        //    }
        //    return newMatrix;
        //}

        //public double[,] ToVector(double[] array)
        //{
        //    double[,] vector = new double[1, array.Length];

        //    for (int i = 0; i < array.Length; i++)
        //    {
        //        vector[0, i] = array[i];
        //    }

        //    return vector;
        //}
    }
}
