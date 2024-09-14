
using System;

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

namespace Convolution
{
    class Program
    {
        static void Main(string[] Args)
        {
            Console.WriteLine("Hello World, how are you?");


            // Create a sample input
            double[,,] input = CreateSampleInput();

            // Create the neural network
            cNeural nn = new cNeural((5, 5, 3), 3, 2); // 5x5x3 input, 3x3 kernel, 2 filters

            // Perform forward propagation
            double[,,] output = nn.ForwardPropagation(input);

            // Print the output
            PrintOutput(output);
        }

        static double[,,] CreateSampleInput()
        {
            double[,,] input = new double[3, 5, 5]; // 3 channels, 5x5 image
            Random rand = new Random();
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < 5; i++)
                    for (int j = 0; j < 5; j++)
                        input[c, i, j] = rand.NextDouble();
            return input;
        }

        static void PrintOutput(double[,,] output)
        {
            for (int d = 0; d < output.GetLength(0); d++)
            {
                Console.WriteLine($"Output channel {d}:");
                for (int i = 0; i < output.GetLength(1); i++)
                {
                    for (int j = 0; j < output.GetLength(2); j++)
                    {
                        Console.Write($"{output[d, i, j]:F2} ");
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
        }
    }

    class cNeural
    {
        int input_height;
        int input_width;
        int input_depth;
        double[,,] kernel; // Changed to 3D array to accommodate kernel shape
        double[] biases; // Changed to 1D array for biases



        //this class takes the input shape, size of kernel filter, and the number of kernels
        public cNeural((int, int, int) input_shape, int kernel_size, int depth)
        {
            input_height = input_shape.Item1;
            input_width = input_shape.Item2;
            input_depth = input_shape.Item3;

            // Calculate output shape
            int output_height = input_height - kernel_size + 1;
            int output_width = input_width - kernel_size + 1;

            // Initialize kernel and biases
            var kernel= new double[depth, input_depth, kernel_size, kernel_size]; // 3D array for kernels
            biases = new double[depth]; // 1D array for biases

            InitializeRandomKernel();
        }

        public double[,,] ForwardPropagation(double[,,] input)
        {
            int output_height = input_height - kernel.GetLength(2) + 1;
            int output_width = input_width - kernel.GetLength(2) + 1;
            int output_depth = kernel.GetLength(0);

            double[,,] output_shape = new double[output_depth, output_height, output_width];

            for (int d = 0; d < output_depth; d++)
            {
                for (int i = 0; i < output_height; i++)
                {
                    for (int j = 0; j < output_width; j++)
                    {
                        double sum = 0;
                        for (int c = 0; c < input_depth; c++)
                        {
                            for (int m = 0; m < kernel.GetLength(2); m++)
                            {   
                                for (int n = 0; n < kernel.GetLength(2); n++)
                                {
                                    sum += input[c, i + m, j + n] * kernel[d, c, m * kernel.GetLength(2) + n];
                                }
                            }
                        }
                        output_shape[d, i, j] = sum + biases[d];
                    }
                }
            }

        return output_shape;
    }

        public void InitializeRandomKernel()
        {
            Random random = new Random();
            for (int i = 0; i < kernel.GetLength(0); i++) // Iterate over depth
            {
                for (int j = 0; j < kernel.GetLength(1); j++) // Iterate over input depth
                {
                    for (int k = 0; k < kernel.GetLength(2); k++) // Iterate over kernel size
                    {
                        kernel[i, j, k] = random.NextDouble(); // Initialize kernel with random values
                    }
                }
                biases[i] = random.NextDouble(); // Initialize bias for each filter
            }
        }
    }
}}