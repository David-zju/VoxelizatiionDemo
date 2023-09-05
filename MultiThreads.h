#pragma once
#include <thread>
#include <vector>
#include "Entities.h"
#include "Cell.h"

struct ThreadParams{
    std::vector<std::vector<std::vector<Ohm_slice::Cell>>>* cell_list_ptr;
    std::vector<Entity>* entities_ptr;
    size_t resolution;
    size_t start = 0;
    size_t end = 0;
};

// 定义线程函数的类型
typedef void(*ThreadFunction)(ThreadParams &params);

void InitialMultiThreads(int numThreads, std::vector<std::vector<std::vector<Ohm_slice::Cell>>>& cell_list, 
                        std::vector<Entity>& entities, size_t resolution, ThreadFunction func);
void cell_clash_thread(ThreadParams& params);
void voxelization_thread(ThreadParams& params);
void id_convert(size_t position, size_t dim1, size_t dim2, size_t dim3, size_t& x, size_t& y, size_t& z);
