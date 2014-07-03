#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define LIST_SIZE (100000)
#define MAX_SOURCE_SIZE (0x100000)

static unsigned long time_math_h(void)
{
        int *vec1, *vec2, *res;
        struct timeval start, stop;

        vec1    = malloc(sizeof(int) * LIST_SIZE);
        vec2    = malloc(sizeof(int) * LIST_SIZE);
        res     = calloc(LIST_SIZE, sizeof(int));

        /* check allocation of arrays */
        if (!vec1 || !vec2 || !res) {
                fprintf(stderr, "Couldn't allocate vec1 || vec2 || res\n");
                exit(1);
        }

        /* fill arrays with ints */
        for (int i = 0; i < LIST_SIZE; i++) {
                vec1[i] = i;
                vec2[i] = LIST_SIZE - i;
        }

        gettimeofday(&start, NULL);

        for (int i = 0; i < LIST_SIZE; i++)
                res[i] = vec1[i] + vec2[i];

        gettimeofday(&stop, NULL);

        free(vec1);
        free(vec2);
        free(res);
        
        return stop.tv_usec - start.tv_usec;
}

static unsigned long time_opencl(void)
{
        FILE *kernel_f;
        char *src_str;
        size_t src_size;

        cl_platform_id          platform_id;
        cl_device_id            dev_id;
        cl_uint                 num_devs;
        cl_uint                 num_platforms;
        cl_int                  ret;
        cl_context              context;
        cl_command_queue        cmd_q;

        cl_mem                  vec1_buff;
        cl_mem                  vec2_buff;
        cl_mem                  res_buff;

        cl_kernel               kernel;
        cl_program              prog;

        int *vec1, *vec2, *res;
        struct timeval start, stop;

        /* allocate arrays */
        vec1    = malloc(sizeof(int) * LIST_SIZE);
        vec2    = malloc(sizeof(int) * LIST_SIZE);
        res     = calloc(LIST_SIZE, sizeof(int));

        /* check allocation of arrays */
        if (!vec1 || !vec2 || !res) {
                fprintf(stderr, "Couldn't allocate vec1 || vec2 || res\n");
                exit(1);
        }

        /* fill arrays with ints */
        for (int i = 0; i < LIST_SIZE; i++) {
                vec1[i] = i;
                vec2[i] = LIST_SIZE - i;
        }

        /* load kernel */
        kernel_f = fopen("vector_add_kernel.cl", "r");

        /* make sure kernel loaded */
        if (!kernel_f) {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
        }

        src_str         = malloc(MAX_SOURCE_SIZE);
        src_size        = fread(src_str, 1, MAX_SOURCE_SIZE, kernel_f);
        fclose(kernel_f);

        /* get platform and device information */
        platform_id     = NULL;
        dev_id          = NULL;

        ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
        ret = clGetDeviceIDs(
                platform_id,
                CL_DEVICE_TYPE_DEFAULT,
                1,
                &dev_id,
                &num_devs
        );


        /* create OpenCL context */
        context = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &ret);

        /* create command queue */
        cmd_q   = clCreateCommandQueue(context, dev_id, 0, &ret);

        /* create memory buffers for arrays */
        vec1_buff = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                LIST_SIZE * sizeof(int),
                NULL,
                &ret
        );

        vec2_buff = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                LIST_SIZE * sizeof(int),
                NULL,
                &ret
        );

        res_buff = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                LIST_SIZE * sizeof(int),
                NULL,
                &ret
        );

        /* copy heap allocated buffers to resp cl buffer */
        ret = clEnqueueWriteBuffer(
                cmd_q, vec1_buff, CL_TRUE, 0,
                LIST_SIZE * sizeof(int), vec1, 0, NULL, NULL
        );

        ret = clEnqueueWriteBuffer(
                cmd_q, vec1_buff, CL_TRUE, 0,
                LIST_SIZE * sizeof(int), vec2, 0, NULL, NULL
        );

        /* create program from kernel source */
        prog = clCreateProgramWithSource(
                context, 1, (const char **)&src_str,
                (const size_t *)&src_size, &ret
        );

        /* build program */
        ret = clBuildProgram(prog, 1, &dev_id, NULL, NULL, NULL);

        /* create kernel */
        kernel = clCreateKernel(prog, "vector_add", &ret);

        /* set arguments of kernel */
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&vec1_buff);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&vec2_buff);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&res_buff);

        /* execute OpenCL kernel on the list */
        size_t global_item_size = LIST_SIZE; /* run on entire list */
        size_t local_item_size = 64; /* chunk size */

        /* run on gpu */
        gettimeofday(&start, NULL);
        ret = clEnqueueNDRangeKernel(
                cmd_q, kernel, 1, NULL, &global_item_size,
                &local_item_size, 0, NULL, NULL
        );
        gettimeofday(&stop, NULL);

        ret = clFlush(cmd_q);
        ret = clFinish(cmd_q);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(prog);
        ret = clReleaseMemObject(vec1_buff);
        ret = clReleaseMemObject(vec2_buff);
        ret = clReleaseMemObject(res_buff);
        ret = clReleaseCommandQueue(cmd_q);
        ret = clReleaseContext(context);
        
        free(vec1);
        free(vec2);
        free(res);

        return stop.tv_usec - start.tv_usec;
}


int main(void)
{
        /* assert that I'm not an idiot */
        assert(NULL == 0);

        printf("Adding vectors using math.h: %lu\n", time_math_h());
        printf("Adding vectors using opencl: %lu\n", time_opencl());

        return 0;
}
