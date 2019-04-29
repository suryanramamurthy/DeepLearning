using System;
using System.Threading.Tasks;


// Word index will start from 1
namespace Embeddings
{
    public unsafe class UnsafeEmbeddings
    {
        private float[,] weights;
        private int V, N;
        private float Eta;
        private object[] weightLocks;

        public UnsafeEmbeddings(int v, int n, float eta)
        {
            this.V = v;
            this.N = n;
            this.Eta = eta;
            this.weights = new float[2 * V, N]; // The first half is W<I,H> and second half is W<H,O>
            weightLocks = new object[2 * V];
            intializeWeights(); // Only Glorot-Normal is supported as of now
        }

        private float sigma(float input)
        {
            return (float)(1.0F / (1.0F + Math.Exp(-1.0F * input)));
        }

        private void intializeWeights()
        {
            fixed (float* ptr = &this.weights[0, 0])
            {
                Random rand = new Random();
                float randOffset = (float)(2.0F / Math.Sqrt(V + N));
                float randRange = 2 * randOffset;
                for (int i = 0; i < 2 * V * N; i += 4)
                {
                    *(ptr + i) = (float)(rand.NextDouble() * randRange - randOffset);
                }
            }
        }

        private void trainWeights(int inputWordIndex, int[] contextWordIndex, int[] negSampling)
        {
            float[,] EJ = new float[contextWordIndex.Length + negSampling.Length, N];
            float[] EH = new float[N];
            fixed (float* ptr = &this.weights[0, 0])
            {
                int wIOffset = (inputWordIndex - 1) * N * 4;
                int wpOffset = V * N * 4;
                // for each pair of input and output word index calculate weight updates
                for (int i = 0; i < contextWordIndex.Length; i++)
                {
                    // For context words, the target is 1. The "val" is common for both EJ and EH vectors
                    float val = sigma(cosine(inputWordIndex, contextWordIndex[i])) - 1.0F;
                    for (int j = 0; j < N; j++)
                    {
                        // All word indices start from 1. So subtract 1 to offset from 0.
                        EJ[i, j] = val * (*(ptr + wIOffset + (j * 4)));
                        //EJ[i, j] = val * this.weights[inputWordIndex - 1, j];
                        EH[j] += val * (*(ptr + wpOffset + ((contextWordIndex[i] - 1) * N + j) * 4));
                        //EH[j] += val * this.weights[V + contextWordIndex[i] - 1, j];
                    }
                }
                for (int i = 0; i < negSampling.Length; i++)
                {
                    // For negative sampling words, the target is 0. The "val" is common for both EJ and EH vectors
                    float val = sigma(cosine(inputWordIndex, negSampling[i]));
                    for (int j = 0; j < N; j++)
                    {
                        // All word indices start from 1. So subtract 1 to offset from 0.
                        EJ[contextWordIndex.Length + i, j] = val * (*(ptr + wIOffset + (j * 4)));
                        //EJ[contextWordIndex.Length + i, j] = val * this.weights[inputWordIndex - 1, j];
                        EH[j] += val * (*(ptr + wpOffset + ((negSampling[i] - 1) * N + j) * 4));
                        //EH[j] += val * this.weights[V + negSampling[i] - 1, j];
                    }
                }
                // Updated W' weights for context words
                Parallel.For(0, contextWordIndex.Length, i =>
                {
                    lock (this.weightLocks[contextWordIndex[i] - 1])
                    {
                        for (int j = 0; j < N; j++)
                        {
                            //*(ptr + wIOffset + j * 4) -= this.Eta * EJ[i, j];
                            this.weights[V + contextWordIndex[i] - 1, j] -= this.Eta * EJ[i, j];
                        }
                    }
                });

                Parallel.For(0, negSampling.Length, i =>
                {
                    lock (this.weightLocks[negSampling[i] - 1])
                    {
                        for (int j = 0; j < N; j++)
                            this.weights[V + negSampling[i] - 1, j] -= this.Eta * EJ[contextWordIndex.Length + i, j];
                    }
                });


                // Update W weights for the input words
                lock (this.weightLocks[inputWordIndex - 1])
                {
                    for (int i = 0; i < N; i++)
                        this.weights[inputWordIndex - 1, i] -= this.Eta * EH[i];
                }
            }
        }

        private float cosine(int inputWordIndex, int outputWordIndex)
        {
            float val = 0;
            fixed (float* ptr = &this.weights[0, 0])
            {
                int iOffset = (inputWordIndex - 1) * N * 4;
                int oOffset = (V + outputWordIndex - 1) * N * 4;
                for (int i = 0; i < N; i += 4)
                    val += (*(ptr + iOffset + i)) * (*(ptr + oOffset + i));
            }
            return val;
        }
    }
}
