#include "stl_reader.h"
#include "json_reader.h"
#include "Entities.h"
#include "Cell.h"
#include <iostream>
#include <vector>
#include "MultiThreads.h"
#include "File.h"
#include <chrono>

int main() {
    std::string json_file = "../models/run1.json";
    std::string stl_file = "../models/run1.stl";
    const int resolution = 64;

    size_t numThreads = std::thread::hardware_concurrency();
    std::cout << "Number of supported threads: " << numThreads << std::endl;
    
    // set the id and the real position of the cells
    std::vector<std::vector<std::vector<Ohm_slice::Cell>>> cell_list = Ohm_slice::Cell::build_cell_list(json_file);

    // set the triangles(normal included) and the material info
    std::vector<Entity> entities;
    load_entities(json_file, stl_file, entities);
    int save_id = 0;
    // entities.erase(entities.begin(), entities.begin()+save_id);
    // entities.erase(entities.begin()+9, entities.end());
    for(size_t i = 0; i < entities.size(); i++){
        std::cout<<i<<" "<< entities[i].triangles.size() <<std::endl;
    }
//     std::cout << entities.size() << std::endl;
    // build the BVH tree
    for(size_t i = 0; i < entities.size(); i++){
        entities[i].build_BVH();
    }
    // entities.erase(entities.begin()+1, entities.end());
    std::cout<< "BVH trees are built." << std::endl;
    
    int sum = 0;
    std::cout<<"Starting matching cells with clashed entities..."<<std::endl;
    auto start  = std::chrono::high_resolution_clock::now();
    std::vector<std::tuple<size_t, size_t, size_t>> cell_id_list;
    for(size_t x = 0; x < cell_list.size(); x++){
        for(size_t y = 0; y < cell_list[0].size(); y++){
            for(size_t z = 0; z < cell_list[0][0].size(); z++){
                cell_id_list.push_back(std::make_tuple(x,y,z));
            }
        }
    }
    InitialMultiThreads(numThreads, cell_list, cell_id_list, entities, resolution, cell_clash_thread);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time taken: " << duration/1.0E6 << " s" << std::endl;
    std::cout<<"cells with clashed entities are matched."<<std::endl;
    for(size_t x = 0; x < cell_list.size(); x++){
        for(size_t y = 0; y < cell_list[0].size(); y++){
            for(size_t z = 0; z < cell_list[0][0].size(); z++){
                if(cell_list[x][y][z].clash_lst.size() > 0) sum++;
            }
        }
    }
    std::cout <<"total_clash_cell_sum = "<< sum << std::endl;
    // // Voxelization with raycasting locally
    // for(size_t x = 0; x < cell_list.size(); x++){
    //     for(size_t y = 0; y < cell_list[0].size(); y++){
    //         for(size_t z = 0; z < cell_list[0][0].size(); z++){
    //             if(cell_list[x][y][z].clash_lst.size()>0){
    //                 cell_list[x][y][z].refine_to_voxels(resolution);
    //                 cell_list[x][y][z].raycast(entities);
    //             }
    //         }
    //     }
    // }
    std::cout<< "Starting dexelization..." <<std::endl;
    start  = std::chrono::high_resolution_clock::now();
    cell_id_list.clear();
    for(size_t x = 0; x < cell_list.size(); x++){
        for(size_t y = 0; y < cell_list[0].size(); y++){
            for(size_t z = 0; z < cell_list[0][0].size(); z++){
                if(cell_list[x][y][z].clash_lst.size() > 0){
                    cell_id_list.push_back(std::make_tuple(x,y,z));
                }
            }
        }
    }
    InitialMultiThreads(numThreads, cell_list, cell_id_list, entities, resolution, voxelization_thread);
    // InitialMultiThreads(numThreads, cell_list, entities, resolution, cell_clash_thread);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time taken: " << duration/1.0E6 << " s" << std::endl;
    // coordinate steps = coordinate(2,2,2) * (1.0/resolution);
    // for(size_t x = 0; x < resolution; x++){
    //     for(size_t y = 0; y < resolution; y++){
    //         // Ray ray(coordinate(x+0.5,y+0.5,0).times(steps) + coordinate(-1,-1,-1));
    //         // Status status;
    //         // double t = ray.Intersect(entities[1], status);
    //         // if(t != std::numeric_limits<double>::max()){
    //         //     std::cout<< "x, y = " << x <<", "<< y << "  \n" <<ray.origin + t*coordinate(0,0,1);
    //         // }
    //     }
    // }
    // Ray ray(coordinate(8+0.5,13+0.5,0).times(steps) + coordinate(-1,-1,-1));
    // Status status;
    // double t = ray.Intersect(entities[1], status);
    // if(t != std::numeric_limits<double>::max()){
    //     std::cout<< "x, y = " << 6 <<", "<< 13 << "  \n" <<ray.origin + t*coordinate(0,0,1);
    // }
    // for(size_t i = 0; i<entities[1].triangles.size(); i++){
    //     // Status status;
    //     double t = ray.Intersect(entities[1].triangles[i], status);
    //     if(t != std::numeric_limits<double>::max()){
    //         std::cout<< entities[1].triangles[i] <<std::endl;
    //         std::cout<< "i = " << i<<"  \n" <<ray.origin + t*coordinate(0,0,1);
    //     }
    // }

    // for (int i = 0; i < resolution; ++i) {
    //     std::cout << "Layer " << i << ":\n";
    //     for (int j = 0; j < resolution; ++j) {
    //         for (int k = 0; k < resolution; ++k) {
    //             bool b1 = cell_list[0][0][0].dexels->dexels[0][i][j].isInside((k+0.5)/(float)resolution);
    //             std::cout<<(int)(b1) <<" ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n";
    // }

    // for (int i = 0; i < resolution; ++i) {
    //     std::cout << "Layer " << i << ":\n";
    //     for (int j = 0; j < resolution; ++j) {
    //         for (int k = 0; k < resolution; ++k) {
    //             bool b1 = cell_list[0][0][0].voxels->voxel_list[k][j][i]->inner_lst.size() > 0;
    //             bool b2 = cell_list[0][0][0].voxels->voxel_list[k][j][i]->clash_lst.size() > 0;
    //             std::cout<<(int)(b2||b1) <<" ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n";
    // }

    // Ray ray(coordinate(0.234,0,-11.234));
    // coordinate collision, collision_;
    // Status status;
    // double t = ray.Intersect(entities[1], status);
    // // // for(triangle tri : entities[1].triangles){
    // // //     T_NUMBER t = ray.Intersect(tri);
    // // //     std::cout<< t <<" "<<tri.debug_id <<" "<< ray.origin + t * coordinate(0,0,1) <<std::endl;
    // // // }
    // std::cout<< ray.origin + t * coordinate(0,0,1)<<status<<std::endl;
    // ray.origin = ray.origin + (t+ACCURACY_THRESHOD) * coordinate(0,0,1);
    // t = ray.Intersect(entities[1], status);
    // std::cout<< ray.origin + t * coordinate(0,0,1)<<status<<std::endl;
    // double step = 1E-10;
    // for(int i = 0; collision.z != collision_.z; step = step/2){
    //     ray.origin = collision + coordinate(0,0,1E-11);
    //     // std::cout<<ray.Intersect(entities[1], collision_)<<" "<< collision <<std::endl;
    //     std::cout<< collision <<collision_;
    // }
    // std::cout<<std::scientific << step<<std::endl;
    
    // Ohm_slice::Cell cell(coordinate(0,0,0), coordinate(16,16*2,16*3), 0, 0, 0);
    // cell.refine_to_voxels(16);
    // std::cout<< *cell.voxels->voxel_list[3][2][1]<< std::endl;
    // Box box(coordinate(0,0,0), coordinate(1,1,1));
    // triangle tri(coordinate(1.5,1.5,0), coordinate(0,1,1.01), coordinate(0,0,3.01));
    // std::cout<<box.clash(tri)<<std::endl;
    
    
    // Ohm_slice::Cell cell(coordinate(0,0,0), coordinate(1,1,1), 0,0,0);
    // Voxels voxels(4, cell);
    return 0;
}
