using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Examples
{
    public class LayeredNeuralNetwork
    {
        public class NeuronLayer
        {
            Matrix<double> synaptic_weights { get; set; }
            //public Matrix<double> SynapticWeights => synaptic_weights;
            public Matrix<double> SynapticWeights
            {
                get { return synaptic_weights; }
                set { synaptic_weights = value; }
            }

            int mNumberOfNeurons { get; set; }
            int mNumberOfInputsPerNeuron { get; set; }

            public NeuronLayer(int numberOfNeurons, int numberOfInputsPerNeuron)
            {
                mNumberOfInputsPerNeuron = numberOfNeurons;
                mNumberOfInputsPerNeuron = numberOfInputsPerNeuron;

                Random random = new Random();

                synaptic_weights = Matrix<double>.Build.Dense(numberOfInputsPerNeuron,
                    numberOfNeurons, (int arg1, int arg2) => (random.NextDouble() * 2) - 1);
            }
        }

        //NeuronLayer mLayer1 { get; set; }
        //public NeuronLayer Layer1 { get { return mLayer1; } }
        //NeuronLayer mLayer2 { get; set; }
        //public NeuronLayer Layer2 { get { return mLayer2; } }

        //public LayeredNeuralNetwork(NeuronLayer layer1, NeuronLayer layer2)
        //{
        //    mLayer1 = layer1;
        //    mLayer2 = layer2;
        //}

        public NeuronLayer[] OrderedLayers { get; set; }

        public LayeredNeuralNetwork(params NeuronLayer[] orderedLayers)
        {
            OrderedLayers = orderedLayers;
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
                Matrix <double> outputLayer_error = expectedOutputs - outputLayer;
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

                Matrix<double> previousSynapticWeights = OrderedLayers.Last().SynapticWeights;
                Matrix<double> previousLayerDelta = outputLayer_delta;


                List<Matrix<double>> orderedLayerAdjustments = new List<Matrix<double>>();
                orderedLayerAdjustments.Add(thisLayerAdjustment);

                for (int i = OrderedLayers.Count() - 2; i >= 0; i--)
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

                    previousSynapticWeights = OrderedLayers[i].SynapticWeights;
                    previousLayerDelta = thisLayer_delta;
                }

                orderedLayerAdjustments.Reverse();

                for (int i = 0; i < OrderedLayers.Count(); i++)
                {
                    NeuronLayer layer = OrderedLayers[i];
                    layer.SynapticWeights += orderedLayerAdjustments[i];
                }

                //// calculate error for layer 2
                //Matrix<double> layer2_error = expectedOutputs - outputFromLayer2;

                //double[] layer2_deltaArray = new double[layer2_error.RowCount];
                //for (int i = 0; i < layer2_error.RowCount; i++)
                //{
                //    layer2_deltaArray[i] = layer2_error[i, 0] * SigmoidDerivative(outputFromLayer2[i, 0]);
                //}
                //Matrix<double> layer2_delta = Matrix<double>.Build.Dense(layer2_error.RowCount, 1, layer2_deltaArray);

                //// calculate error for layer 1 (by looking at the weights in layer 1, 
                //// we can determine by how much layer 1 contributed to the error in layer 2
                //Matrix<double> layer1_error = layer2_delta * mLayer2.SynapticWeights.Transpose();

                //double[,] layer1_deltaArray = new double[layer1_error.RowCount, layer1_error.ColumnCount];
                //for (int i = 0; i < layer1_error.RowCount; i++)
                //{
                //    for (int j = 0; j < layer1_error.ColumnCount; j++)
                //    {
                //        layer1_deltaArray[i, j] = layer1_error[i, j] * SigmoidDerivative(outputFromLayer1[i, j]);
                //    }
                //}
                //Matrix<double> layer1_delta = Matrix<double>.Build.Dense(layer1_error.RowCount, layer1_error.ColumnCount, (x, y) => layer1_deltaArray[x, y]);

                //// calculate how much to adjust the weights by
                //Matrix<double> layer1_adjustment = inputs.Transpose() * layer1_delta;
                //Matrix<double> layer2_adjustment = outputFromLayer1.Transpose() * layer2_delta;

                //// adjust the weights
                //this.mLayer1.SynapticWeights += layer1_adjustment;
                //this.mLayer2.SynapticWeights += layer2_adjustment;
            }
        }

        //public void Think(Matrix<double> input, out Matrix<double> outputFromLayer1, out Matrix<double> outputFromLayer2)
        //{
        //    outputFromLayer1 = (input * this.mLayer1.SynapticWeights).Map(o => this.Sigmoid(o));
        //    outputFromLayer2 = (outputFromLayer1 * this.mLayer2.SynapticWeights).Map(o => this.Sigmoid(o));
        //}

        public void Think(Matrix<double> input, out List<Matrix<double>> orderedOutputs)
        {
            orderedOutputs = new List<Matrix<double>>();

            Matrix<double> inputToNextLayer = input;

            foreach (NeuronLayer layer in this.OrderedLayers)
            {
                Matrix<double> outputFromThisLayer = (inputToNextLayer * layer.SynapticWeights).Map(o => this.Sigmoid(o));
                orderedOutputs.Add(outputFromThisLayer);
                inputToNextLayer = outputFromThisLayer;
            }
        }
    }
}
