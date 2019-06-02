#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

size_t round_n(int n, int block_size)
{
    return ((size_t)(n + block_size - 1) / block_size) * block_size;
}

void add_to_blocks(double* a, double* b, double *c, int N,
                   const cl::Program& program, const cl::Context& context, const cl::CommandQueue& queue, int block_size) {
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * N);
    cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * N);
    cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * N);

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N, a);
    queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * N, b);

    size_t range = round_n(N, block_size);

    cl::Kernel kernel(program, "add_to_blocks");
    cl::KernelFunctor add_to_blocks(kernel, queue, cl::NullRange, cl::NDRange(range), cl::NDRange(block_size));
    add_to_blocks(dev_a, dev_c, dev_b, N);
    queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * N, c);
}

void block_prefix_sum(double* a, double* b, double* sums, int N,
                      const cl::Program& program, const cl::Context& context, const cl::CommandQueue& queue,
                      int block_size) {
    // allocate device buffer to hold message
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(double) * N);
    cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(double) * N);

    // copy from cpu to gpu
    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N, a);

    // load named kernel from opencl source
    size_t range = round_n(N, block_size);
    cl::Kernel kernel(program, "scan_hillis_steele");
    cl::KernelFunctor prefix_sum(kernel, queue, cl::NullRange,
                                 cl::NDRange(range), cl::NDRange(block_size));
    prefix_sum(dev_a, dev_b, cl::__local(sizeof(double) * block_size), cl::__local(sizeof(double) * block_size), N);

    queue.enqueueReadBuffer(dev_b, CL_TRUE, 0, sizeof(double) * N, b);

    if (block_size < N)
    {
        int blocks = N / block_size + 1;
        double current_sum = 0;
        for (int i = 1; i < blocks; ++i)
        {
            current_sum += b[block_size * i - 1];
            sums[i] = current_sum;
        }

        add_to_blocks(b, sums, b, N, program, context, queue, block_size);
    }
}

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("prefix_sum.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        size_t const block_size = 256;
        try
        {
            program.build(devices, "-DBLOCK_SIZE=256");
        }
        catch (cl::Error const & e)
        {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        // create a message to send to kernel
        std::ifstream input_stream("input.txt");
        int N;
        input_stream >> N;

        auto a = new double[N];
        for (int i = 0; i < N; i++)
        {
            input_stream >> a[i];
        }
        auto b = new double[N];
        memset(b, 0, sizeof(double) * N);
        auto sums = new double[N];
        memset(sums, 0, sizeof(double) * N);

        block_prefix_sum(a, b, sums, N, program, context, queue, block_size);


        std::ofstream output_stream("output.txt");
        output_stream << std::fixed << std::setprecision(3);
        for (int i = 0; i < N; i++)
        {
            output_stream << b[i] << " ";
        }

        output_stream.close();
        delete[] sums;
        delete[] b;
        delete[] a;
    }
    catch (cl::Error& e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}