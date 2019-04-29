using System;
using System.Threading.Tasks;
using Distributions;


// Word index will start from 1
// Add method to read and save weights to a file
namespace Embeddings
{
    public class Embeddings
    {
        // hid_weights is from input to hidden layer
        // out_weights is from hidden to output layer
        private float[,] hid_weights, out_weights;
        private int V, N; // V - Vocabulary Size, N - embedding size
        private float Eta; // Learning rate
        // locks for parallelizing the training process
        private object[] hid_weightLocks, out_weightLocks;
        private Zipfian zipf; // Zipfian distribution to select negative samples. 

        /*
         * Parameters:
         * v --> vocabulary size
         * n --> Embeddings Dimension
         * initial --> True for creating random weights. False, will read the weightsFile to initialize the weights
         * weightsFile --> The file in binary format containing the embeddings parameters to be loaded
         */
        public Embeddings(int v, int n, float eta, bool initial, string weightsFile)
        {
            this.V = v + 1; // The given word index will start from 1. 0 is reserved for 'UNK'
            this.N = n;
            this.Eta = eta;
            this.hid_weights = new float[V, N];
            this.out_weights = new float[N, V];
            this.hid_weightLocks = new object[V];
            this.out_weightLocks = new object[V];
            this.zipf = new Zipfian(1.0F, this.V); // Zipfian with skew 1 and for vocab size this.V
            if (initial)
                intializeWeights(); // Only Glorot-Normal is supported as of now
                                    //else 
                                    // call the method to read weights from a checkpoint file
        }

        private float sigma(float input)
        {
            return (float)(1.0F / (1.0F + Math.Exp(-1.0F * input)));
        }

        private void intializeWeights()
        {
            Random rand = new Random();
            float randOffset = (float)(2.0F / Math.Sqrt(this.V + this.N));
            float randRange = 2 * randOffset;
            for (int i = 0; i < this.V; i++)
                for (int j = 0; j < this.N; j++)
                    this.hid_weights[i, j] = (float)(rand.NextDouble() * randRange - randOffset);
            for (int i = 0; i < this.N; i++)
                for (int j = 0; j < this.V; j++)
                    this.out_weights[i, j] = (float)(rand.NextDouble() * randRange - randOffset);
        }

        /*
         * This method returns an array of sampleSize with word indices that needs to be reinforced in the negative
         * context. It will ensure that the word indices are unique and are not part of the context words that needs
         * to be positively reinforced. This method assumes that a mini-batch is being provided for training. If no
         * mini-batch then the first dimension value should be 1. 
         */
        private int[] generateNegativeSamples(int sampleSize, int[,] contextWords)
        {
            int[] negSamples = new int[sampleSize];
            Random rand = new Random();
            int i = 0;
            while (i < sampleSize)
            {
                double p = rand.NextDouble();
                long wordIndex = this.zipf.zipfInvCDF(p);
                bool inContext = false;
                // Check if the random wordIndex is not part of the context word index list
                for (int j = 0; j < contextWords.GetLength(0); j++)
                    for (int k = 0; k < contextWords.GetLength(1); k++)
                        if (contextWords[j, k] == wordIndex) inContext = true;
                // Check if the random wordIndex has not been already included in the negSamples array
                for (int j = 0; j < i; j++)
                    if (wordIndex == negSamples[j]) inContext = true;
                if (!inContext)
                {
                    negSamples[i] = (int)wordIndex;
                    i++;
                }
            }
            return negSamples;
        }


        /*
        * The inputwordindex and the contextwordindex list will start from 1. 0 is reserved for 'UNK'
        * The parameters for this method are
        * 1: batchSize - The number of training samples in this mini-batch
        * 2: inputWordIndex - Input word to be trained. 1D array. The length denotes the batch size
        * 3: contextWordIndex - 2D array of context word indices. Dimension 0 is batch and 1 is context words 
        * 4: negSamples - Number of negative samples to use for this batch. The negSamples chosen
        *                      will be ensured that they are not part of the context word Index
        * 
        * Current implementation does not support batchSize. It will be implemented in the next version
        */

        private float trainWeights(int batchSize, int[] inputWordIndex, int[,] contextWordIndex, int negSample)
        {
            // Ensure that the dimensions for inputWordIndex and contextWordIndex are the same as batchSize
            if (inputWordIndex.Length != batchSize)
                throw new ArgumentOutOfRangeException("The batchsize should equal the number of input words");
            if (contextWordIndex.GetLength(0) != batchSize)
                throw new ArgumentOutOfRangeException("The batchsize should equal the zeroeth dimension of context words");

            // Create negative samples
            float loss = 0;
            int[] negSamples = generateNegativeSamples(negSample, contextWordIndex);

            // EJC is the gradients for the output weights w.r.t. the context words.
            // EJN is the gradients for the output weights w.r.t. the negative sample words.
            // Since the input word for each batch can be different, the weights have to updated for the gradient
            // for each batch cumulatively. Since the context and input word indices can vary from batch to batch
            // the gradients are also stored in the batch dimension.
            float[,,] EJC = new float[batchSize, contextWordIndex.GetLength(1), N];
            // EH is the gradients for the hidden weights w.r.t. the input words.
            float[,,] EJN = new float[batchSize, negSample, N];
            float[,] EH = new float[batchSize, N];

            // Per batch, calculate the output weight gradients for the context words.
            for (int batch = 0; batch < batchSize; batch++)
                for (int i = 0; i < contextWordIndex.GetLength(1); i++) // For the number of context words in that batch
                {
                    // For context words, the target is 1. The "val" is common for both EJ and EH vectors
                    float temp = sigma(cosine(inputWordIndex[batch], contextWordIndex[batch, i]));
                    float val = temp - 1.0F;
                    loss += (float)-Math.Log(temp); // Loss computation
                    for (int j = 0; j < N; j++)
                    {
                        EJC[batch, i, j] += val * this.hid_weights[inputWordIndex[batch], j];
                        // EH also has to be fixed. Since there are multiple input words based on batch size
                        EH[batch, j] += val * this.out_weights[j, contextWordIndex[batch, i]];
                    }
                }
            // Per batch, calculate the output weight gradients for the negative sample words.
            for (int batch = 0; batch < batchSize; batch++)
                for (int i = 0; i < negSamples.Length; i++)
                {
                    // For negative sampling words, the target is 0. The "val" is common for both EJ and EH vectors
                    float temp = sigma(cosine(inputWordIndex[batch], negSamples[i]));
                    loss += (float)-Math.Log(-temp);
                    for (int j = 0; j < N; j++)
                    {
                        EJN[batch, i, j] = temp * this.hid_weights[inputWordIndex[batch], j];
                        EH[batch, j] += temp * this.out_weights[j, negSamples[i]];
                    }
                }

            for (int batch = 0; batch < batchSize; batch++)
                // Updated W' weights for context words
                Parallel.For(0, contextWordIndex.GetLength(1), i =>
                {
                    lock (this.out_weightLocks[contextWordIndex[batch, i]])
                    {
                        for (int j = 0; j < N; j++)
                        {
                            this.out_weights[j, contextWordIndex[batch, i]] -= this.Eta * EJC[batch, i, j];
                        }
                    }
                });

            for (int batch = 0; batch < batchSize; batch++)
                Parallel.For(0, negSamples.Length, i =>
                {
                    lock (this.out_weightLocks[negSamples[i]])
                    {
                        for (int j = 0; j < N; j++)
                            this.out_weights[j, negSamples[i]] -= this.Eta * EJN[batch, i, j];
                    }
                });


            for (int batch = 0; batch < batchSize; batch++)
                // Update W weights for the input words
                lock (this.hid_weightLocks[inputWordIndex[batch]])
                {
                    for (int i = 0; i < N; i++)
                        this.hid_weights[inputWordIndex[batch], i] -= this.Eta * EH[batch, i];
                }
            return loss;
        }

        private float cosine(int inputWordIndex, int outputWordIndex)
        {
            float val = 0;
            for (int i = 0; i < N; i++)
                val += this.hid_weights[inputWordIndex, i] * this.out_weights[i, outputWordIndex];
            return val;
        }
    }
}
