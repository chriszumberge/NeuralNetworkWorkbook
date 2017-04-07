using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.DecisionTree
{
    public class Instance
    {
        public List<Feature> Features { get; set; } = new List<Feature>();
        public Output Output { get; set; }

        public Instance Split(string axis, string value)
        {
            var splitFeature = new Feature(value, axis);
            var featureSplit = Features.Where(f => !f.Equals(splitFeature)).ToList();

            // no split happened
            if (featureSplit.Count == Features.Count)
            {
                featureSplit = new List<Feature>();
            }

            return new Instance
            {
                Output = Output,
                Features = featureSplit
            };
        }

        public override string ToString()
        {
            var s = Features.Aggregate(String.Empty, (acc, item) => acc + item.Value + ", ");
            s += Output.Value;

            return s;
        }
    }
}
