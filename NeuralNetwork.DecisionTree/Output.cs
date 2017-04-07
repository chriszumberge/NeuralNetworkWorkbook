using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.DecisionTree
{
    [DataContract]
    public class Output : Feature
    {
        public Output(string value, string @class) : base(value, @class) { }
    }
}
