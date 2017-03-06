using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuronLayer
    {
        // The number of neurons in this layer
        int mNumNeurons { get; set; }
        public int NumNeurons => mNumNeurons;

        // The layer of neurons
        List<Neuron> mNeurons = new List<Neuron>();
        public List<Neuron> Neurons { get { return mNeurons; } }

        public NeuronLayer(Random random, int numNeurons, int numInputsPerNeuron)
        {
            mNumNeurons = numNeurons;
            for (int i = 0; i < mNumNeurons; i++)
            {
                mNeurons.Add(new Neuron(random, numInputsPerNeuron));
            }
        }
    }
}
