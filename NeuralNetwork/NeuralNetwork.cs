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

        //public NeuralNetwork(Random random, int numInputs, int numOutputs, params int[] numHiddenLayerNeurons)        
        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNetwork"/> class.
        /// </summary>
        /// <param name="random">The random.</param>
        /// <param name="numInputs">The number inputs.</param>
        /// <param name="numNeuronsFirstLayer">The first layer. Output layer in networks with no hidden layers</param>
        /// <param name="numHiddenLayerNeurons">The number hidden layer neurons.</param>
        public NeuralNetwork(Random random, int numInputs, int numNeuronsFirstLayer, params int[] numNeuronsAdditionalLayers)
        {
            // Handle bias
            mNumInputs = numInputs;

            List<int> consecutiveLayers = new int[] { numNeuronsFirstLayer }.Concat(numNeuronsAdditionalLayers).ToList();
            
            mNumOuputs = consecutiveLayers.Last();
            consecutiveLayers.RemoveAt(consecutiveLayers.Count - 1);

            int numLastOutputs = mNumInputs;

            // Add all the hidden layers connecting the previous output to this input
            for (int i = 0; i < consecutiveLayers.Count; i++)
            {
                mLayers.Add(new NeuronLayer(random, consecutiveLayers[i], numLastOutputs));
                numLastOutputs = consecutiveLayers[i];
            }

            // Add the output layer
            mLayers.Add(new NeuronLayer(random, mNumOuputs, numLastOutputs));
        }

        public NeuralNetwork(Random random, int numInputs, params int[] numNeuronsAdditionalLayers)
        {
            if (numNeuronsAdditionalLayers.Length < 1)
            {
                throw new ArgumentException("Must be at least one additional layer for output");
            }

            mNumInputs = numInputs;

            List<int> consecutiveLayers = numNeuronsAdditionalLayers.ToList();

            mNumOuputs = consecutiveLayers.Last();
            consecutiveLayers.RemoveAt(consecutiveLayers.Count - 1);

            int numLastOutputs = numInputs;

            // Add all the hidden layers connecting the previous output to this input
            for (int i = 0; i < consecutiveLayers.Count; i++)
            {
                mLayers.Add(new NeuronLayer(random, consecutiveLayers[i], numLastOutputs));
                numLastOutputs = consecutiveLayers[i];
            }

            // Add the output layer
            mLayers.Add(new NeuronLayer(random, mNumOuputs, numLastOutputs));
        }

        public List<int> GetNumberOfNeuronsPerLayer()
        {
            List<int> num = new List<int>();
            num.Add(mNumInputs);
            foreach (NeuronLayer layer in mLayers)
            {
                num.Add(layer.NumberOfNeurons);
            }
            return num;
        }

        //The Sigmoid function, normalizes between 0 and 1, the higher the response modifier the more steep the curve
        private double Sigmoid(double x, double responseModifier = 1.0)
        {
            return 1 / (1 + Math.Exp(-x / responseModifier));
        }

        // Derivative of the Sigmoid funciton.
        // The Gradient of the Sigmoid curve.
        // It indicates how confident we are about the existing weight
        private double SigmoidDerivative(double x)
        {
            return (x) * (1 - (x));
        }

        public void Train(Matrix<double> inputs, Matrix<double> expectedOutputs, int numberOfTrainingIterations)
        {
            // Add identity for bias
            //inputs = inputs.InsertColumn(inputs.ColumnCount, Vector<double>.Build.Dense(inputs.RowCount, 1));

            for (int t = 0; t < numberOfTrainingIterations; t++)
            {
                List<Matrix<double>> outputs = new List<Matrix<double>>();
                this.Think(inputs, mLayers, out outputs);

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
        
        private void Think(Matrix<double> input, List<NeuronLayer> layers, out List<Matrix<double>> orderedOutputs)
        {
            orderedOutputs = new List<Matrix<double>>();

            Matrix<double> inputToNextLayer = input;

            foreach (NeuronLayer layer in layers)
            {
                // Modify for bias
                Matrix<double> weights = layer.SynapticWeights;
                //for (int colCount = 0; colCount < weights.ColumnCount; colCount++)
                //{
                //    weights[weights.RowCount - 1, colCount] *= Parameters.Bias;
                //}

                Matrix<double> outputFromThisLayer = (inputToNextLayer * weights).Map(o => this.Sigmoid(o, Parameters.ActivationResponse));
                orderedOutputs.Add(outputFromThisLayer);
                inputToNextLayer = outputFromThisLayer;
            }
        }

        public void Think(Matrix<double> input, out List<Matrix<double>> orderedOutputs)
        {
            //input = input.InsertColumn(input.ColumnCount, Vector<double>.Build.Dense(input.RowCount, 1));

            Think(input, mLayers, out orderedOutputs);
        }

        public void Think(Vector<double> input, out List<Matrix<double>> orderedOutputs)
        {
            Matrix<double> matrixInput = input.ToRowMatrix();
            //matrixInput = matrixInput.InsertColumn(matrixInput.ColumnCount, Vector<double>.Build.Dense(matrixInput.RowCount, 1));

            this.Think(matrixInput, out orderedOutputs);
        }
    }

    public static class Parameters
    {
        public static int Bias = -1;
        public static int ActivationResponse = 1;
    }
}
