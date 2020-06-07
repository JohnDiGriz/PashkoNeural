using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;

namespace PashkoNeural
{
    public class NeuralNetwork
    {
        private Random _rand;
        public double[] _weights;
        public NeuralNetwork(int inputs)
        {
            _rand = new Random(1);
            _weights = new double[inputs];
            for (int i = 0; i < inputs; i++)
            {
                _weights[i] = 2 * _rand.NextDouble() - 1;
            }
        }
        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        private double[][] T(double[][] a)
        {
            var res = new double[a[0].Length][];
            for (int i = 0; i < a[0].Length; i++)
            {
                res[i] = new double[a.Length];
            }
            for (int j = 0; j < a.Length; j++)
            {
                for (int i = 0; i < a[0].Length; i++)
                {
                    res[i][j] = a[j][i];
                }
            }
            return res;
        }
        private double ScalarMultiplication(double[] x, double[] y)
        {
            return x.Select((a, i) => a * y[i]).Sum();
        }
        private double[] Multiplication(double x, double[] y)
        {
            return y.Select(a => a * x).ToArray();
        }
        private double[] Multiplication(double[] x, double[] y)
        {
            return x.Select((a, i) => a * y[i]).ToArray();
        }
        private double DerivativeSigmoid(double x)
        {
            return x * (1 - x);
        }
        public void Train(double[][] trainingInputs, double[] trainingOutputs, int nIterations)
        {
            for (int i = 0; i < nIterations; i++)
            {
                var output = Think(trainingInputs);
                var error = trainingOutputs.Select((x, i) => x - output[i]).ToArray();
                var adjustment = T(trainingInputs)
                    .Select(x => ScalarMultiplication(x, 
                        Multiplication(error, 
                            output.Select(x => DerivativeSigmoid(x)).ToArray()
                            )
                        ))
                    .ToArray();
                for(int j=0;j<adjustment.Length;j++)
                {
                    _weights[j] += adjustment[j];
                }
            }
        }
        public double[] Think(double[][] inputs)
        {
            return inputs.Select(x => Sigmoid(ScalarMultiplication(x, _weights))).ToArray();
        }
        public double Think(double[] inputs)
        {
            return Sigmoid(ScalarMultiplication(inputs, _weights));
        }
    }
}
