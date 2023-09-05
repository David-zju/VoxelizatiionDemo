#include "MultiThreads.h"
#include "Cell.h"
#include "Entities.h"
#include "my_macros.h"
#include <iostream>
#include <unistd.h>

void cell_clash_thread(ThreadParams& params){
    std::vector<std::vector<std::vector<Ohm_slice::Cell>>>& cell_list = (*params.cell_list_ptr);
    std::vector<Entity>& entities = (*params.entities_ptr);
    size_t start = params.start;
    size_t end = params.end;
    size_t resolution = params.resolution;

    size_t dim1 = cell_list.size();
    size_t dim2 = cell_list[0].size();
    size_t dim3 = cell_list[0][0].size();
    // size_t totalSize = dim1 * dim2 * dim3;
    size_t x,y,z;
    for(size_t id = start; id < end; id++){
        id_convert(id, dim1, dim2, dim3, x, y, z);
        for(size_t id = 0; id < entities.size(); id++){
            // 注意短路原则，metal判断放前面
            if(entities[id].is_metal && cell_list[x][y][z].clash(entities[id])) cell_list[x][y][z].clash_lst.push_back(id);
        }
    }
}

void voxelization_thread(ThreadParams& params){
    std::vector<std::vector<std::vector<Ohm_slice::Cell>>>& cell_list = (*params.cell_list_ptr);
    std::vector<Entity>& entities = (*params.entities_ptr);
    size_t start = params.start;
    size_t end = params.end;
    size_t resolution = params.resolution;

    size_t dim1 = cell_list.size();
    size_t dim2 = cell_list[0].size();
    size_t dim3 = cell_list[0][0].size();
    // size_t totalSize = dim1 * dim2 * dim3;
    size_t x,y,z;
    for(size_t id = start; id < end; id++){
        id_convert(id, dim1, dim2, dim3, x, y, z);
        if(cell_list[x][y][z].clash_lst.size()>0){
            // cell_list[x][y][z].refine_to_voxels(resolution);
            // cell_list[x][y][z].raycast_voxel(entities);
            cell_list[x][y][z].refine_to_dexels(resolution);
            cell_list[x][y][z].raycast_dexel(entities);
        }
    }
}

void InitialMultiThreads(int numThreads, std::vector<std::vector<std::vector<Ohm_slice::Cell>>>& cell_list, 
                        std::vector<Entity>& entities, size_t resolution, ThreadFunction func){
    std::vector<std::thread> threads;
    size_t dim1 = cell_list.size();
    size_t dim2 = cell_list[0].size();
    size_t dim3 = cell_list[0][0].size();
    size_t totalSize = dim1 * dim2 * dim3;

    size_t partitionSize = totalSize/numThreads;
    size_t start, end;
    std::vector<ThreadParams> params_lst(numThreads);
    for(size_t i = 0; i < numThreads; i++){
        params_lst[i].cell_list_ptr = &cell_list;
        params_lst[i].entities_ptr = &entities;
        params_lst[i].resolution = resolution;
        params_lst[i].start = i * partitionSize;
        params_lst[i].end = (i == numThreads - 1) ? cell_list.size() * cell_list[0].size() * cell_list[0][0].size() : ((i+1) * partitionSize);
        threads.emplace_back(func, std::ref(params_lst[i]));
        // sleep(0.5);
        #ifdef ENABLE_PRINT
            std::cout<<"Thread with ID: "<< std::this_thread::get_id() <<" Created."<<std::endl;
        #endif
    }
    for(std::thread& threadObj : threads){
        threadObj.join();
        #ifdef ENABLE_PRINT
            std::cout<<"Thread with ID: "<< std::this_thread::get_id() <<" Finished."<<std::endl;
        #endif
    }
    #ifdef ENABLE_PRINT
        std::cout << "All threads have finished." << std::endl;
    #endif
}

void id_convert(size_t position, size_t dim1, size_t dim2, size_t dim3, size_t& x, size_t& y, size_t& z){
    // 坐标转换
    x = position / (dim2 * dim3);
    y = (position % (dim2 * dim3)) / dim3;
    z = position % dim3;
}
