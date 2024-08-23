/*
The code in the present file is for the greatest part based on the work of
Lars Björn van den Haak which can be found at
https://github.com/sakehl/gpu-bisimulation,
which is based on the paper
Jan Martens, Jan Friso Groote, Lars B. van den Haak, Pieter Hijma,
and Anton Wijs. A Linear Parallel Algorithm to Compute Bisimulation and
Relational Coarsest Partitions. Formal Aspects of Component Software (Facs),
2021.
*/

#include <iostream>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/limits>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "Bisimulation.h"

namespace mlconcepts {

inline static void gpuAssertInternal(cudaError_t code, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "CUDA Assert: " << cudaGetErrorString(code);
        std::cerr << " (line " <<  line << ")";
        std::cerr << std::endl;
        if (abort) exit(code);
    }
}

#define gpuAssert(code) gpuAssertInternal(code, __LINE__)

/*
    Preprocessing Code Starts here
*/
struct thrust_incr : public thrust::unary_function<int,int>
{
    __host__ __device__
    int operator()(int x) { return x+1; }
};

template<typename NodeID, typename LabelID>
__global__ void mark_p(NodeID M, NodeID* sources, LabelID* labels, bool* marks,
                       LabelID current_action, NodeID* blocks)
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M && labels[i] == current_action) {
        marks[sources[i]] = true;
    }
}

template<typename NodeID>
__global__ void mark_p_nolabels(NodeID M, NodeID* sources, bool* marks,
                                NodeID* blocks)
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M) {
        marks[sources[i]] = true;
    }
}

template<typename NodeID>
__global__ void leaderElect_p(NodeID N, bool* marks, NodeID* blocks,
                              NodeID* next_numbers)
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && marks[blocks[i]] != marks[i]) {
        next_numbers[blocks[i]] = i;
    }
}

template<typename NodeID>
__global__ void split_p(NodeID N, bool* marks, NodeID* blocks, 
                        NodeID* next_numbers)
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && marks[blocks[i]] != marks[i]) {
        blocks[i] = next_numbers[blocks[i]];
    }
}

template<typename NodeID, typename LabelID>
void make_partition(NodeID N, NodeID M, LabelID L, NodeID* sources,
                    NodeID* targets, LabelID* labels, NodeID* blocks)
{
    bool* marks;
    NodeID* next_numbers;
    gpuAssert(cudaMalloc((void **) &marks, N * sizeof(bool)));
    gpuAssert(cudaMalloc((void **) &next_numbers, N * sizeof(NodeID)));

    //Setup threads
    constexpr int threads_M = 32;
    int blocks_M = (M + threads_M -1) / threads_M;
    constexpr int threads_N = 32;
    int blocks_N = (N + threads_N -1) / threads_N;
    
    if (L < 2) {
        cudaMemset(marks, 0, N * sizeof(bool));
        mark_p_nolabels<<<blocks_M, threads_M>>>(M, sources, marks, blocks);
        leaderElect_p<<<blocks_N, threads_N>>>(N, marks, blocks, next_numbers);
        split_p<<<blocks_N, threads_N>>>(N, marks, blocks, next_numbers);
    } else {
        //Go over each action label
        for(LabelID a = 0; a < L; ++a){
            //Reset the marks
            cudaMemset(marks, 0, N * sizeof(bool));

            mark_p<<<blocks_M, threads_M>>>(M, sources, labels, 
                                            marks, a, blocks);
            leaderElect_p<<<blocks_N, threads_N>>>(N, marks, blocks, 
                                                   next_numbers);
            split_p<<<blocks_N, threads_N>>>(N, marks, blocks, next_numbers);
        }
    }

    gpuAssert(cudaFree(next_numbers));
    gpuAssert(cudaFree(marks));
}

//Calculates action switch
template <typename NodeID, typename LabelID>
__global__ void calc_switch(NodeID M, NodeID* sources, LabelID* labels, 
                            NodeID* action_switch)
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < M) {
        if(i == 0 || sources[i] != sources[i - 1] || labels[i] == labels[i-1]){
            action_switch[i] = 0;
        } else{
            action_switch[i] = 1;
        }
    }
}

template <typename NodeID, typename LabelID>
void sort_transitions(NodeID M, LabelID L, NodeID* sources, NodeID* targets,
                      LabelID* labels)
{
    if (L > 1) {
        thrust::device_ptr<NodeID> sources_th(sources);
        thrust::device_ptr<NodeID> targets_th(targets);
        thrust::device_ptr<LabelID> labels_th(labels);

        auto first = thrust::make_zip_iterator(thrust::make_tuple(sources_th,
                                                                labels_th,
                                                                targets_th));
        auto last  = thrust::make_zip_iterator(thrust::make_tuple(sources_th + M,
                                                                labels_th + M,
                                                                targets_th + M));
        thrust::sort(first, last);
    } else {
        thrust::device_ptr<NodeID> sources_th(sources);
        thrust::device_ptr<NodeID> targets_th(targets);

        auto first = thrust::make_zip_iterator(thrust::make_tuple(sources_th,
                                                                targets_th));
        auto last  = thrust::make_zip_iterator(thrust::make_tuple(sources_th + M,
                                                                targets_th + M));
        thrust::sort(first, last);
    }
}

//Given the transistions (sources and labels), fill the order,
//nr_mark and mark_offset on the device and return total marks_length
//N is the number states, M is the number of transitions.
template <typename NodeID, typename LabelID>
int make_marks_offset(NodeID N, NodeID M, LabelID L, NodeID* sources,
                      LabelID* labels, NodeID* order, NodeID* nr_marks,
                      NodeID* marks_offset)
{
    int threads_M = 32;
    int blocks_M = (M + threads_M -1) / threads_M;
    //Make the action_switch array (size M) on the device
    NodeID* action_switch_d;
    gpuAssert(cudaMalloc((void **) &action_switch_d, M * sizeof(NodeID)));

    //Call kernel calc_switch_flags to calculate flags and action_switch
    //(total work M)
    if (L < 2) {
        gpuAssert(cudaMemset(action_switch_d, 0, M * sizeof(NodeID)));
    } else {
        calc_switch<<<blocks_M, threads_M>>>(M, sources, labels,
                                             action_switch_d);
    }

    //Do a segmented inclusive prefix sum (scan) to calculate order (thrust)
    thrust::device_ptr<NodeID> action_switch_th(action_switch_d);
    thrust::device_ptr<NodeID> order_th(order);
    thrust::device_ptr<NodeID> sources_th(sources);

    thrust::inclusive_scan_by_key(sources_th, sources_th + M,
                                  action_switch_th, order_th);

    //Initialize nr_marks to zeros
    //With a reduce and scatter from thrust, we can calculate the nr_marks
    gpuAssert(cudaMemset(nr_marks, 0, N * sizeof(NodeID)));
    thrust::device_ptr<NodeID> nr_marks_th(nr_marks);
    thrust::device_ptr<NodeID> marks_offset_th(marks_offset);

    thrust::device_ptr<NodeID> keys = thrust::device_malloc(N * sizeof(NodeID));
    thrust::device_ptr<NodeID> values = thrust::device_malloc(N * sizeof(NodeID));

    auto new_end = thrust::reduce_by_key(sources_th, sources_th + M,
                                         action_switch_th, keys, values);

    thrust::transform(values, new_end.second, values, thrust_incr());
    thrust::scatter(values, new_end.second, keys, nr_marks_th);

    //Do an exclusive prefix sum on nr_marks to calculate marks_offset (thrust)
    thrust::exclusive_scan(nr_marks_th, nr_marks_th + N, marks_offset_th);

    //Calculate marks_length from marks_offset and nr_marks
    NodeID marks_length = marks_offset_th[N-1] + nr_marks_th[N-1];
    gpuAssert(cudaFree(action_switch_d) );

    return marks_length;
}



template<typename NodeID=std::uint32_t, typename LabelID=std::uint32_t>
class LTSDevice {
public:
    LTSDevice(const LTS<NodeID, LabelID>& _lts,
              std::vector<NodeID>& blocks) : lts(_lts) {
        init_device(blocks);
    }

    ~LTSDevice() {
        free_device();
    }

    void preprocess() {
        NodeID N = lts.GetNodesSize();
        NodeID M = lts.GetEdgesSize();
        LabelID L = lts.GetLabelsCount();
        sort_transitions(M, L, source_d, target_d, label_d);
        make_partition(N, M, L, source_d, target_d, label_d, block_d);
        marks_length = make_marks_offset(N, M, L, source_d, label_d, order_d,
                                         nr_mark_d, marks_offset_d);
    }

    NodeID* source_d;
    NodeID* target_d;
    LabelID* label_d;
    NodeID* order_d;

    NodeID* block_d;
    NodeID* nr_mark_d;
    NodeID* marks_offset_d;
    NodeID marks_length;
private:
    const LTS<NodeID, LabelID>& lts;

    void init_device(std::vector<NodeID>& blocks) {
        std::size_t N = lts.GetNodesSize();
        std::size_t M = lts.GetEdgesSize();
        gpuAssert(cudaMalloc((void **) &source_d, M * sizeof(NodeID)));
        if (lts.IsLabelled()) {
            gpuAssert(cudaMalloc((void **) &label_d, M * sizeof(NodeID)));
        }
        gpuAssert(cudaMalloc((void **) &target_d, M * sizeof(NodeID)));

        gpuAssert(cudaMalloc((void **) &order_d, M * sizeof(NodeID)));
        gpuAssert(cudaMalloc((void **) &nr_mark_d, N * sizeof(NodeID)));
        gpuAssert(cudaMalloc((void **) &marks_offset_d, N * sizeof(NodeID)));
        gpuAssert(cudaMalloc((void **) &block_d, N * sizeof(NodeID)));

        gpuAssert(cudaMemcpy((void**)block_d, blocks.data(), 
                             N * sizeof(NodeID), cudaMemcpyHostToDevice));

        gpuAssert(cudaMemcpy(source_d, lts.GetSources().data(),
                             sizeof(NodeID) * M, cudaMemcpyHostToDevice));
        if (lts.IsLabelled()) {
            gpuAssert(cudaMemcpy(label_d, lts.GetLabels().data(),
                                 sizeof(NodeID) * M, cudaMemcpyHostToDevice));
        }
        gpuAssert(cudaMemcpy(target_d, lts.GetTargets().data(),
                             sizeof(NodeID) * M, cudaMemcpyHostToDevice));
    }

    void free_device() {
        gpuAssert(cudaFree(source_d));
        if (lts.IsLabelled())
            gpuAssert(cudaFree(label_d));
        gpuAssert(cudaFree(target_d));

        gpuAssert(cudaFree(order_d));
        gpuAssert(cudaFree(nr_mark_d));
        gpuAssert(cudaFree(marks_offset_d));
    }
};

// Step 0
template<typename NodeID>
__global__ void set_stable(int N, bool* stable, NodeID* block) {
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        stable[block[i]] = false;
    }
}

//Step 1: reset values and pick a block
template<typename NodeID>
__global__ void pick_block(NodeID N, bool* stable, bool* mark, 
                           NodeID* current_block) {
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    // Reset the markings of previous round.
    if(i < N) {
        mark[i] = false;
        if(!stable[i]) {
            *current_block = i;
        }
   }
}

// Step 2: Mark the states which can reach the current block && set the current
// block to stable
template<typename NodeID>
__global__ void mark(int M, NodeID* source, NodeID* target, NodeID* order,
                     NodeID* marks_offset, bool* stable, bool* marks,
                     NodeID* current_block, NodeID* block) 
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < M) {
        if(block[target[i]] == *current_block) {
            // Represents the transition source[i] ->labels[i] target[i]
            marks[marks_offset[source[i]] + order[i]] = true;
       }
    }

    //Set current block to stable
    if (i < 1 && *current_block != cuda::std::numeric_limits<NodeID>::max())
        stable[*current_block] = true;
}

template<typename NodeID> 
__device__ NodeID customCAS(NodeID* ptr, NodeID cmp, NodeID val) {
    return atomicCAS(ptr, cmp, val);
}

template<>
__device__ std::uint64_t customCAS(std::uint64_t* ptr, std::uint64_t cmp, 
                                   std::uint64_t val) {
    return atomicCAS((unsigned long long int*)ptr, (unsigned long long int)cmp, 
                     (unsigned long long int)val);
}


// Step 3: Check for every transition if markings between leader is different
// and elect it as a new leader
template<typename NodeID>
__global__ void compare_markings(NodeID M, NodeID* source, NodeID* order,
                                 NodeID* marks_offset, bool* mark, bool* marks,
                                 NodeID* block, NodeID* next_number)
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M){
        if( marks[marks_offset[      source[i] ] + order[i]] != 
            marks[marks_offset[block[source[i]]] + order[i]]) {

            mark[source[i]] = true;
            next_number[block[source[i]]] = source[i];
        }
    }
}

// Step 4: Split the block, update the block of the split off states
template<typename NodeID>
__global__ void split(NodeID N, bool* stable, bool* mark, NodeID* block,
                      NodeID* next_number, NodeID* current_block)
{
    NodeID i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && mark[i]) {
        stable[block[i]] = false;
        block[i] = next_number[block[i]];
        stable[block[i]] = false;
        stable[*current_block] = false;
    }
}

template<typename NodeID, typename LabelID>
static int ComputeBisimulationCudaInternal(const LTS<NodeID, LabelID>& lts,
                                           std::vector<NodeID>& blocks)
{
    NodeID N = lts.GetNodesSize();
    NodeID M = lts.GetEdgesSize();
    NodeID L = lts.GetLabelsCount();
    LTSDevice<NodeID, LabelID> lts_d(lts, blocks);

    lts_d.preprocess();

    //Setting the block sizes (threads per block) and nr of block
    int threads_N = 32;
    int blocks_N = (N + threads_N -1) / threads_N;

    int threads_M = 32;
    int blocks_M = (M + threads_M -1) / threads_M;
    
    //All states have a mark and marks array, also it has the block number
    bool *mark_d;
    bool *marks_d;
    gpuAssert( cudaMalloc((void**)&mark_d, sizeof(bool) * N) );
    gpuAssert(cudaMalloc((void**)&marks_d, sizeof(bool) * lts_d.marks_length));

    // All block have a next_number (which is the next leader)
    // and indicate if they are stable
    NodeID* next_number_d;
    bool* stable_d;
    gpuAssert( cudaMalloc((void**)&next_number_d, sizeof(int) * N) );
    gpuAssert( cudaMalloc((void**)&stable_d, sizeof(bool) * N) );

    gpuAssert( cudaMemset(stable_d, 1, sizeof(bool) * N) );
    set_stable<<<blocks_N, threads_N>>>(N, stable_d, lts_d.block_d);

    //The current block, undefined (-1) in the beginning
    NodeID c = std::numeric_limits<NodeID>::max();
    NodeID *c_d;
    gpuAssert( cudaMalloc((void**)&c_d, sizeof(int)) );
    gpuAssert( cudaMemcpy(c_d, &c, sizeof(int), cudaMemcpyHostToDevice) );

    int iter = 0;
    // Executing kernel
    do {
        iter++;
        //Set current block to undefined
        c = std::numeric_limits<NodeID>::max();
        gpuAssert( cudaMemcpy(c_d, &c, sizeof(NodeID), cudaMemcpyHostToDevice) );

        // Step1: Pick the block to split
        pick_block<<<blocks_N, threads_N>>>(N, stable_d, mark_d, c_d);
        //Step 1a: reset marks
        gpuAssert( cudaMemset(marks_d, 0, sizeof(bool) * lts_d.marks_length) );

        //Loop over the transitions to mark with the current block.
        mark<<<blocks_M, threads_M>>>(M, lts_d.source_d, lts_d.target_d, 
                                      lts_d.order_d, lts_d.marks_offset_d, 
                                      stable_d, marks_d, c_d, lts_d.block_d);

        //Compare markings and elect new leaders
        compare_markings<<<blocks_M, threads_M>>>(M, lts_d.source_d, 
                                lts_d.order_d, lts_d.marks_offset_d, mark_d,
                                marks_d, lts_d.block_d, next_number_d);

        //Split of the marked block
        split<<<blocks_N, threads_N>>>(N, stable_d, mark_d, lts_d.block_d,
                                       next_number_d, c_d);

        //Get back the current block
        gpuAssert( cudaMemcpy(&c, c_d, sizeof(int), cudaMemcpyDeviceToHost) );
    } while( c != std::numeric_limits<NodeID>::max() && iter < 10*N );
  
    if(c != -1){
        std::cerr << "WARNING: We passed a reasonable number of iterations("
                  << iter << "), but we are not stable yet." << std::endl;
        return -1;
    }

    gpuAssert(cudaMemcpy(blocks.data(), lts_d.block_d, sizeof(NodeID) * N,
                         cudaMemcpyDeviceToHost) );

    gpuAssert( cudaPeekAtLastError() );
    gpuAssert( cudaDeviceSynchronize() );

    // Deallocate device memory
    gpuAssert( cudaFree(marks_d) );
    gpuAssert( cudaFree(mark_d) );
    gpuAssert( cudaFree(next_number_d) );
    gpuAssert( cudaFree(stable_d) );
    gpuAssert( cudaFree(c_d) );

    return iter;
}

template<typename NodeID, typename LabelID>
void ComputeBisimulationCuda(const LTS<NodeID, LabelID>& lts,
                             std::vector<NodeID>& blocks)
{
    ComputeBisimulationCudaInternal(lts, blocks);
}

// Instantiates the bisimulation computation function.

template void ComputeBisimulationCuda<std::uint32_t, std::uint8_t>(
                                const LTS<std::uint32_t, std::uint8_t>& lts,
                                std::vector<std::uint32_t>& blocks);

template void ComputeBisimulationCuda<std::uint32_t, std::uint16_t>(
                                const LTS<std::uint32_t, std::uint16_t>& lts,
                                std::vector<std::uint32_t>& blocks);

template void ComputeBisimulationCuda<std::uint32_t, std::uint32_t>(
                                const LTS<std::uint32_t, std::uint32_t>& lts,
                                std::vector<std::uint32_t>& blocks);

template void ComputeBisimulationCuda<std::uint32_t, std::uint64_t>(
                                const LTS<std::uint32_t, std::uint64_t>& lts,
                                std::vector<std::uint32_t>& blocks);

template void ComputeBisimulationCuda<std::uint64_t, std::uint8_t>(
                                const LTS<std::uint64_t, std::uint8_t>& lts,
                                std::vector<std::uint64_t>& blocks);

template void ComputeBisimulationCuda<std::uint64_t, std::uint16_t>(
                                const LTS<std::uint64_t, std::uint16_t>& lts,
                                std::vector<std::uint64_t>& blocks);

template void ComputeBisimulationCuda<std::uint64_t, std::uint32_t>(
                                const LTS<std::uint64_t, std::uint32_t>& lts,
                                std::vector<std::uint64_t>& blocks);

template void ComputeBisimulationCuda<std::uint64_t, std::uint64_t>(
                                const LTS<std::uint64_t, std::uint64_t>& lts,
                                std::vector<std::uint64_t>& blocks);

}