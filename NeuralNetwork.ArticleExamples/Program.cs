using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.ArticleExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine($"Finding Local Minima for f(x,y) = 1 + x^2 + y^2, starting at (x,y) = (1, 1) with a step size of 0.5");
            Console.Write("> ");
            FindLocalMinimaForFunctionUsingGradient((inputs) =>
            {
                return Vector<double>.Build.DenseOfArray(
                    new double[]
                    {
                        2 * inputs[0],
                        2 * inputs[1]
                    });
            }, Vector<double>.Build.DenseOfArray(new double[] { 1, 1 }), 0.5);

            Console.WriteLine();

            Console.WriteLine($"Finding Local Minima for f(x) = x^2 - 1, starting at x = -1 with a step size of 1.0");
            Console.Write("> ");
            FindLocalMinimaForFunctionUsingGradient((inputs) =>
            {
                return Vector<double>.Build.DenseOfArray(
                    new double[]
                    {
                        2 * inputs[0]
                    });
            }, Vector<double>.Build.DenseOfArray(new double[] { -1 }), 1.0);

            Console.WriteLine();

            Console.WriteLine($"Finding Local Minima for f(x) = x^2 - 1, starting at x = -1 with a step size of 0.25");
            Console.Write("> ");
            FindLocalMinimaForFunctionUsingGradient((inputs) =>
            {
                return Vector<double>.Build.DenseOfArray(
                    new double[]
                    {
                        2 * inputs[0]
                    });
            }, Vector<double>.Build.DenseOfArray(new double[] { -1 }), 0.25);

            Console.WriteLine();

            Console.WriteLine($"Finding the boundary line for (-1, 1) = 1; (0, -1) = -1; (3/2, 1) = 1 without b");
            Console.Write("> ");
            FindBoundaryLine(new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] {-1, 1}),
                Vector<double>.Build.DenseOfArray(new double[] {0, -1}),
                Vector<double>.Build.DenseOfArray(new double[] {3.0/2.0, 1})
            }, new List<double> { 1, -1, 1 }, 2, false);

            Console.WriteLine();

            Console.WriteLine($"Finding the boundary line for (-1, 1) = 1; (0, -1) = -1; (3/2, 1) = 1");
            Console.Write("> ");
            FindBoundaryLine(new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] {-1, 1}),
                Vector<double>.Build.DenseOfArray(new double[] {0, -1}),
                Vector<double>.Build.DenseOfArray(new double[] {3.0/2.0, 1})
            }, new List<double> { 1, -1, 1 }, 2);

            Console.WriteLine();

            Console.WriteLine($"Finding the boundary line for (-1, 1) = 1; (0, -1) = -1; (10, 1) = 1");
            Console.Write("> ");
            FindBoundaryLine(new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] {-1, 1}),
                Vector<double>.Build.DenseOfArray(new double[] {0, -1}),
                Vector<double>.Build.DenseOfArray(new double[] {10, 1})
            }, new List<double> { 1, -1, 1 }, 2);

            Console.ReadLine();
        }

        static void FindLocalMinimaForFunctionUsingGradient(Func<Vector<double>,Vector<double>> functionGradient, Vector<double> initialLocation, double step, double minimumDelta = 0)
        {
            // Xn-1 = Xn - p(gradientF(Xn))
            
            Vector<double> location = initialLocation;

            List<Vector<double>> locations = new List<Vector<double>>()
            {
                location
            };

            bool continueIterating = true;

            while(continueIterating)
            {
                Vector<double> nextLocation = location - (step * functionGradient(location));
                locations.Add(nextLocation);

                if (nextLocation.Equals(location))
                {
                    // We've reached the local minima
                    continueIterating = false;
                    Console.WriteLine($"Local minima found at {nextLocation.PrintVector(false)} after {locations.Count - 1} steps.");
                }
                // We just added nextLocation to the list so we expect the index to be one minus the count of members in the list,
                // however, if the indexOf evaluates to something different, that means that it found the same exact value earlier.. meaning we've
                // circled around and it'll continue to evaluate the loop infinitely
                else if (locations.IndexOf(nextLocation) != locations.Count - 1)
                {
                    // We're unstable, oscillating around the local minima never to reach it
                    continueIterating = false;
                    List<Vector<double>> unstableLocations = locations.GetRange(locations.IndexOf(nextLocation), locations.Count - 1 - locations.IndexOf(nextLocation));
                    string unstableLocationStrings = String.Join(", ", unstableLocations.Select(l => l.PrintVector(false)));
                    Console.WriteLine($"Unstable minima found after {locations.Count - 1} steps around locations {unstableLocationStrings}");
                }
                else if (Math.Abs((nextLocation - location).Sum()) < minimumDelta)
                {
                    // Step size is so small that we've *effectively* reached the local minima but the math may not actually
                    // get us there... good approximation
                    continueIterating = false;
                    Console.WriteLine($"Local minima approximated at {nextLocation.PrintVector(false)} after {locations.Count - 1} steps.");
                }

                location = nextLocation;
            }
        }

        static void FindBoundaryLine(List<Vector<double>> inputs, List<double> outputs, int inputSize, bool includeB = true)
        {
            // Yi = (w^k & Xi) + b^k
            // w^(k+1) = w^k + YiXi
            // b^(k+1) = b^k + Yi

            int dataIndex = 0;
            int consecutiveCorrect = 0;
            int timesIncorrect = 0;

            double b = 0;
            Vector<double> w = Vector<double>.Build.Dense(inputSize, 0.0);

            while (consecutiveCorrect != inputs.Count)
            {
                Vector<double> input = inputs[dataIndex];
                double output = outputs[dataIndex];

                double calculatedOutput = 0.0;
                if (includeB)
                {
                    calculatedOutput = (w * input) + b;
                }
                else
                {
                    calculatedOutput = (w * input);
                }

                // if the calculated output is 0 or the calculated output and expected output don't have the same sign, need to adjust values
                if (calculatedOutput == 0 || ((output > 0 && calculatedOutput < 0) || (output < 0 && calculatedOutput > 0)))
                {
                    consecutiveCorrect = 0;
                    timesIncorrect++;
                    //w = w + (output * input);
                    w += (output * input);

                    if (includeB)
                    {
                        b += output;
                    }
                }
                else
                {
                    consecutiveCorrect++;
                }

                dataIndex++;
                if (dataIndex >= inputs.Count)
                {
                    dataIndex = 0;
                }
            }

            if (includeB)
            {
                Console.WriteLine($"Boundary Vector found at w = {w.PrintVector(false)}, b = {b}, after {timesIncorrect} recalculations.");
            }
            else
            {
                Console.WriteLine($"Boundary Vector found at w = {w.PrintVector(false)}, after {timesIncorrect} recalculations.");
            }
        }
    }

    public static class ExtensionsAndHelpers
    {
        public static string PrintVector(this Vector<double> vector, bool vertical = true)
        {
            StringBuilder sb = new StringBuilder();
            if (!vertical)
            {
                sb.Append("( ");
            }
            for (int i = 0; i < vector.Count; i++)
            {
                if (vertical)
                {
                    sb.AppendLine(vector[i].ToString());
                }
                else
                {
                    sb.Append(vector[i].ToString() + " ");
                }
            }
            if (!vertical)
            {
                sb.Append(")");
            }
            return sb.ToString();
        }
    }
}
