using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Neuron
    {
        // The number of inputs into the neuron
        int mNumInputs { get; set; }
        public int NumInputs => mNumInputs;

        // The weights for each input
        List<double> mWeights { get; set; } = new List<double>();
        public List<double> Weights => mWeights;

        public Neuron(Random random, int numInputs)
        {
            // we need an additional weight for the bias
            mNumInputs = numInputs + 1;

            for (int i = 0; i < mNumInputs; i++)
            {
                // set up the weights with random values between -1 and 1
                mWeights.Add((random.NextDouble() * 2) - 1);
            }
        }

    }
}
