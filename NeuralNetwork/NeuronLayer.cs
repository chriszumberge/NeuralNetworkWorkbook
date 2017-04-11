using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
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

        public int NumberOfNeurons { get { return mNumberOfNeurons;} }
        int mNumberOfNeurons { get; set; }
        int mNumberOfInputsPerNeuron { get; set; }

        public NeuronLayer(Random random, int numberOfNeurons, int numberOfInputsPerNeuron)
        {
            
            mNumberOfNeurons = numberOfNeurons;

            // one additional weight for the bias
            //mNumberOfInputsPerNeuron = numberOfInputsPerNeuron + 1;
            mNumberOfInputsPerNeuron = numberOfInputsPerNeuron;

            synaptic_weights = Matrix<double>.Build.Dense(mNumberOfInputsPerNeuron,
                numberOfNeurons, (int arg1, int arg2) => (random.NextDouble() * 2) - 1);
        }
    }
}
