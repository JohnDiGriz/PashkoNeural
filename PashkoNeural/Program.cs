using System;
using System.Linq;

namespace PashkoNeural
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new NeuralNetwork(3);

            Console.WriteLine("Random starting synaptic weights: ");
            Console.WriteLine(String.Join(", ", network._weights));

            var trainingInputs = new double[4][]
                {
                    new double[3]{ 0, 0, 1 },
                    new double[3]{ 1, 1, 1 },
                    new double[3]{ 1, 0, 1 },
                    new double[3]{ 0, 1, 1 }
                };
            var trainingOutputs = new double[] { 1, 0, 0, 1 };

            network.Train(trainingInputs, trainingOutputs, 1000);

            Console.WriteLine("New synaptic weights after training: ");
            Console.WriteLine(String.Join(", ", network._weights));

            Console.WriteLine("Considering new situation [1, 1, 0] -> ?: ");
            Console.WriteLine(network.Think(new double[] { 1, 1, 0 }));
        }
    }
}
