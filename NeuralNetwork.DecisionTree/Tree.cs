using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.DecisionTree
{
    [DataContract]
    [KnownType(typeof(Feature))]
    [KnownType(typeof(Output))]
    [KnownType(typeof(Tree))]
    public class Tree
    {
        [DataMember]
        public Output Leaf { get; set; }

        [DataMember]
        public Dictionary<Feature, Tree> Branches { get; set; }

        public string DisplayTree(int tab = 0)
        {
            StringBuilder sb = new StringBuilder();

            Action tabWriter = () => Enumerable.Range(0, tab).ToList().ForEach(i => sb.Append("\t"));

            if (Branches != null)
            {
                foreach(var feature in Branches)
                {
                    tabWriter();
                    sb.AppendLine($"if {feature.Key.ToString()}");
                    sb.Append(feature.Value.DisplayTree(tab + 1));
                }
            }
            else
            {
                tabWriter();
                sb.AppendLine(Leaf.ToString());
            }
            return sb.ToString();
        }

        public static Output ProcessInstance(Tree tree, Instance i)
        {
            if (tree.Leaf != null)
            {
                return tree.Leaf;
            }

            return ProcessInstance(tree.TreeForInstance(i), i);
        }

        private Tree TreeForFeature(Feature feature)
        {
            Tree found;
            if (Branches.TryGetValue(feature, out found))
            {
                return found;
            }
            return null;
        }

        private Tree TreeForInstance(Instance instance)
        {
            var tree = instance.Features.Select(f => TreeForFeature(f)).FirstOrDefault(f => f != null);
            return tree;
        }
    }
}
