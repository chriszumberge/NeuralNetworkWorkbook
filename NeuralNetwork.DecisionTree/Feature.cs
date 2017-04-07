using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.DecisionTree
{
    [DataContract]
    public class Feature
    {
        [DataMember]
        public string Value { get; set; }

        [DataMember]
        public string Axis { get; set; }

        public Feature(string value, string axis)
        {
            Value = value;
            Axis = axis;
        }


        protected bool Equals(Feature other)
        {
            return String.Equals(Value, other.Value) && String.Equals(Axis, other.Axis);
        }

        // override object.Equals
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return this.Equals((Feature)obj);
        }

        // override object.GetHashCode
        public override int GetHashCode()
        {
            unchecked
            {
                return ((Value != null ? Value.GetHashCode() : 0) * 397) ^ (Axis != null ? Axis.GetHashCode() : 0);
            }
        }

        public override string ToString()
        {
            return $"{Axis}: {Value}";
        }
    }
}
