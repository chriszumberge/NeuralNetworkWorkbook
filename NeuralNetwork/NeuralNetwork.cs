using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        int mNumInputs { get; set; }
        int mNumOutputs { get; set; }
        int mNumHiddenLayers { get; set; }
        int mNeuronsPerHiddenLayer { get; set; }

        List<NeuronLayer> mLayers { get; set; } = new List<NeuronLayer>();

        Random mRandom { get; set; }

        public NeuralNetwork(Random random)
        {
            mRandom = random;
        }

        public void CreateNetwork()
        {
            if (mNumHiddenLayers > 0)
            {
                // Add the first layer using the inputs
                mLayers.Add(new NeuronLayer(mRandom, mNeuronsPerHiddenLayer, mNumInputs));

                for (int i = 0; i < mNumHiddenLayers - 1; i++)
                {
                    mLayers.Add(new NeuronLayer(mRandom, mNumOutputs, mNeuronsPerHiddenLayer));
                }

                // create output layer
                mLayers.Add(new NeuronLayer(mRandom, mNumOutputs, mNeuronsPerHiddenLayer));
            }
            else
            {
                // create output layer
                mLayers.Add(new NeuronLayer(mRandom, mNumOutputs, mNumInputs));
            }
        }

        public List<double> GetWeights()
        {
            // this will hold the weights
            List<double> weights = new List<double>();

            // for each layer
            for (int i = 0; i < mNumHiddenLayers + 1; i++)
            {
                // for each neuron
                for (int j = 0; j < mLayers[i].NumNeurons; j++)
                {
                    // for each weight
                    for (int k = 0; k < mLayers[i].Neurons[j].NumInputs; k++)
                    {
                        weights.Add(mLayers[i].Neurons[j].Weights[k]);
                    }
                }
            }

            return weights;
        }

        public int GetNumberOfWeights()
        {
            int weights = 0;

            for (int i = 0; i < mNumHiddenLayers + 1; i++)
            {
                for (int j = 0; j < mLayers[i].NumNeurons; j++)
                {
                    for (int k = 0; k < mLayers[i].Neurons[j].NumInputs; k++)
                    {
                        weights++;
                    }
                }
            }

            return weights;
        }

        public void PutWeights(List<double> weights)
        {
            int cWeight = 0;

            // for each layer
            for (int i = 0; i < mNumHiddenLayers + 1; i++)
            {
                // for each neuron
                for (int j = 0; j < mLayers[i].NumNeurons; j++)
                {
                    // for each weight
                    for (int k = 0; k < mLayers[i].Neurons[j].NumInputs; k++)
                    {
                        mLayers[i].Neurons[j].Weights[k] = weights[cWeight++];
                    }
                }
            }

            return;
        }

        // calculates outputs from a set of inputs
        public List<double> Update(List<double> inputs)
        {
            // stores the resultant outputs from each layer
            List<double> outputs = new List<double>();

            int cWeight = 0;

            // first check that we have the correct amount of inputs
            if (inputs.Count != mNumInputs)
            {
                // just return an empty list
                return outputs;
            }

            // for each layer..
            for (int i = 0; i < mNumHiddenLayers + 1; i++)
            {
                if (i > 0)
                {
                    inputs = outputs;
                }
                outputs.Clear();

                cWeight = 0;

                // for each neuron sum the (inputs * corresponding weights)
                // throw the total at our sigmund function to get the output
                for (int j = 0; j < mLayers[i].NumNeurons; j++)
                {
                    double netInput = 0;
                    int NumInputs = mLayers[i].Neurons[j].NumInputs;

                    // for each weight
                    for (int k = 0; k < NumInputs - 1; k++)
                    {
                        // sum the weights x inputs
                        netInput += mLayers[i].Neurons[j].Weights[k] * inputs[cWeight++];
                    }

                    // add in the bias
                    netInput += mLayers[i].Neurons[j].Weights[NumInputs - 1] * Params.Bias;

                    // we can store the outputs from each layer as we generate them
                    // the combined activation is first filtered through the sigmoid function
                    //outputs.Add(Sigmoid(netInput, Params.ActivationResponse));
                    outputs.Add(netInput);

                    cWeight = 0;
                }
            }

            return outputs;
        }

        // sigmoid response curve
        internal double Sigmoid(double activation, double response)
        {
            return (1 / (1 + Math.Exp(-activation / response)));
        }
    }
}
