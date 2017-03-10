using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Training
{
    public class TrainingSet
    {
        double mAverageError { get; set; }
        public double AverageError {  get { return mAverageError; } }

        TimeSpan mTrainingTime { get; set; }
        public TimeSpan TrainingTime { get { return mTrainingTime; } }

        NeuralNetwork mNetwork { get; set; }
        public NeuralNetwork Network {  get { return mNetwork; } }

        TimeSpan mAverageProcessingTime { get; set; }
        public TimeSpan AverageProcessingTime { get { return mAverageProcessingTime; } }

        public TrainingSet(NeuralNetwork trainingNetwork, Matrix<double> inputs, Matrix<double> expectedOutputs, int numberOfTrainingIterations)
        {
            Stopwatch trainingWatch = new Stopwatch();

            trainingWatch.Start();

            trainingNetwork.Train(inputs, expectedOutputs, numberOfTrainingIterations);

            trainingWatch.Stop();

            mTrainingTime = trainingWatch.Elapsed;

            double error = 0;
            double totalProcessingMilliseconds = 0;
            Stopwatch processingWatch = new Stopwatch();

            // score correctness
            for (int i = 0; i < inputs.RowCount; i++)
            {
                Vector<double> inputArray = inputs.Row(i);
                Vector<double> expectedOutput = expectedOutputs.Row(i);

                List<Matrix<double>> processingOutput = new List<Matrix<double>>();

                processingWatch.Start();
                trainingNetwork.Think(inputArray, out processingOutput);
                processingWatch.Stop();

                Vector<double> actualOutput = processingOutput.Last().Row(0);

                Vector<double> difference = expectedOutput - actualOutput;

                for (int j = 0; j < difference.ToArray().Length; j++)
                {
                    error += Math.Abs(difference.ToArray()[j]);
                }

                totalProcessingMilliseconds += processingWatch.ElapsedMilliseconds;
                processingWatch.Reset();
            }

            mAverageError = error / inputs.RowCount;
            mAverageProcessingTime = TimeSpan.FromMilliseconds(totalProcessingMilliseconds / inputs.RowCount);
            mNetwork = trainingNetwork;
        }
    }
}
