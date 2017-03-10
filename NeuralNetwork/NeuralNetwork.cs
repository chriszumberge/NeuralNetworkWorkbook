using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public List<NeuronLayer> Layers { get { return mLayers; } }
        List<NeuronLayer> mLayers { get; set; } = new List<NeuronLayer>();

        public int NumInputs { get { return mNumInputs; } }
        public int mNumInputs { get; set; }

        public int NumOutputs { get { return mNumOuputs; } }
        public int mNumOuputs { get; set; }

        public NeuralNetwork(Random random, int numInputs, int numOutputs, params int[] numHiddenLayerNeurons)
        {
            mNumInputs = numInputs;
            mNumOuputs = numOutputs;

            int numLastOutputs = numInputs;
            
            // Add all the hidden layers connecting the previous output to this input
            for (int i = 0; i < numHiddenLayerNeurons.Length; i++)
            {
                mLayers.Add(new NeuronLayer(random, numHiddenLayerNeurons[i], numLastOutputs));
                numLastOutputs = numHiddenLayerNeurons[i];
            }

            // Add the output layer
            mLayers.Add(new NeuronLayer(random, numOutputs, numLastOutputs));
        }

        public List<int> GetNumberOfNeuronsPerLayer()
        {
            List<int> num = new List<int>();
            num.Add(mNumInputs);
            foreach(NeuronLayer layer in mLayers)
            {
                num.Add(layer.NumberOfNeurons);
            }
            return num;
        }

        //The Sigmoid function, normalizes between 0 and 1
        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        // Derivative of the Sigmoid funciton.
        // The Gradient of the Sigmoid curve.
        // It indicates how confident we are about the existing weight
        private double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }

        public void Train(Matrix<double> inputs, Matrix<double> expectedOutputs, int numberOfTrainingIterations)
        //public void Train(List<Vector<double>> inputs, Vector<double> expectedOutputs, int numberOfTrainingIterations)
        {
            for (int t = 0; t < numberOfTrainingIterations; t++)
            {
                //Matrix<double> outputFromLayer1 = Matrix<double>.Build.Sparse(1, 1);
                //Matrix<double> outputFromLayer2 = Matrix<double>.Build.Sparse(1, 1);
                List<Matrix<double>> outputs = new List<Matrix<double>>();
                //this.Think(inputs, out outputFromLayer1, out outputFromLayer2);
                this.Think(inputs, out outputs);

                Matrix<double> outputLayer = outputs.Last();

                // calculate error for output
                Matrix<double> outputLayer_error = expectedOutputs - outputLayer;
                double[,] outputLayer_deltaArray = new double[outputLayer_error.RowCount, outputLayer_error.ColumnCount];
                for (int i = 0; i < outputLayer_error.RowCount; i++)
                {
                    for (int j = 0; j < outputLayer_error.ColumnCount; j++)
                    {
                        outputLayer_deltaArray[i, j] = outputLayer_error[i, j] * SigmoidDerivative(outputLayer[i, j]);
                    }
                }
                Matrix<double> outputLayer_delta = Matrix<double>.Build.Dense(outputLayer_error.RowCount, outputLayer_error.ColumnCount,
                    (x, y) => outputLayer_deltaArray[x, y]);

                Matrix<double> inputsToThisLayer = outputs.Count == 1 ? inputs : outputs[outputs.Count - 2];
                Matrix<double> thisLayerAdjustment = inputsToThisLayer.Transpose() * outputLayer_delta;

                Matrix<double> previousSynapticWeights = mLayers.Last().SynapticWeights;
                Matrix<double> previousLayerDelta = outputLayer_delta;


                List<Matrix<double>> orderedLayerAdjustments = new List<Matrix<double>>();
                orderedLayerAdjustments.Add(thisLayerAdjustment);

                for (int i = mLayers.Count() - 2; i >= 0; i--)
                {
                    Matrix<double> thisLayerOutput = outputs[i];

                    Matrix<double> thisLayer_error = previousLayerDelta * previousSynapticWeights.Transpose();

                    double[,] thisLayer_deltaArray = new double[thisLayer_error.RowCount, thisLayer_error.ColumnCount];
                    for (int j = 0; j < thisLayer_error.RowCount; j++)
                    {
                        for (int k = 0; k < thisLayer_error.ColumnCount; k++)
                        {
                            thisLayer_deltaArray[j, k] = thisLayer_error[j, k] * SigmoidDerivative(thisLayerOutput[j, k]);
                        }
                    }
                    Matrix<double> thisLayer_delta = Matrix<double>.Build.Dense(thisLayer_error.RowCount, thisLayer_error.ColumnCount,
                        (x, y) => thisLayer_deltaArray[x, y]);

                    inputsToThisLayer = i == 0 ? inputs : outputs[i - 1];
                    thisLayerAdjustment = inputsToThisLayer.Transpose() * thisLayer_delta;
                    orderedLayerAdjustments.Add(thisLayerAdjustment);

                    previousSynapticWeights = mLayers[i].SynapticWeights;
                    previousLayerDelta = thisLayer_delta;
                }

                orderedLayerAdjustments.Reverse();

                for (int i = 0; i < mLayers.Count(); i++)
                {
                    NeuronLayer layer = mLayers[i];
                    layer.SynapticWeights += orderedLayerAdjustments[i];
                }
            }
        }

        public void Think(Matrix<double> input, out List<Matrix<double>> orderedOutputs)
        {
            orderedOutputs = new List<Matrix<double>>();

            Matrix<double> inputToNextLayer = input;

            foreach (NeuronLayer layer in mLayers)
            {
                Matrix<double> outputFromThisLayer = (inputToNextLayer * layer.SynapticWeights).Map(o => this.Sigmoid(o));
                orderedOutputs.Add(outputFromThisLayer);
                inputToNextLayer = outputFromThisLayer;
            }
        }

        public void Think(Vector<double> input, out List<Matrix<double>> orderedOutputs)
        {
            this.Think(input.ToRowMatrix(), out orderedOutputs);
        }
    }
}
