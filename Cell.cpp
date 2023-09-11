#include "Cell.h"
#include "my_macros.h"
#include <cmath>
#include <algorithm>
#include <typeinfo>

void Ohm_slice::Cell::refine_to_voxels(int res){
    voxels = new Voxels(res, *this);
}

void Ohm_slice::Cell::refine_to_dexels(int res){
    dexels_x = new Dexels(res, clash_lst);
    dexels_y = new Dexels(res, clash_lst);
    dexels_z = new Dexels(res, clash_lst);
}

coordinate Ohm_slice::Cell::iloc(coordinate realloc){
    using namespace std;
    coordinate relative = realloc - LeftDown();
    size_t resolution = voxels->res();
    coordinate step_lens = steps(resolution, resolution, resolution);
    return coordinate(floor(relative.x/step_lens.x), floor(relative.y/step_lens.y), floor(relative.z/step_lens.z));
}

void Ohm_slice::Cell::raycast_voxel(const std::vector<Entity>& entities){
    // clash_lst.size() > 0 
    // 要和所有的entity求交，并体素化
    size_t resolution = voxels->res();
    coordinate step_lens = steps(resolution, resolution, resolution);
    const size_t z = 0;
    T_NUMBER t = std::numeric_limits<T_NUMBER>::max();
    Status status;
    std::vector<std::pair<size_t, Status>> scan_lst; // 扫描线的交点信息
    for(size_t x = 0; x < resolution; x++){
        for(size_t y = 0; y < resolution; y++){
            for(size_t entity_id = 0; entity_id < clash_lst.size(); entity_id++){
                Ray ray(coordinate(x+0.5, y+0.5, z).times(step_lens) + LeftDown(),Z);
                // x = 6789 y = 13
                // if(entity_id != 1) continue;
                // if(x != 6 || y != 13) continue;
                scan_lst.clear();
                while (true){
                    t = ray.Intersect(entities[clash_lst[entity_id]], status);
                    // no intersection
                    if(t == std::numeric_limits<T_NUMBER>::max()) break;
                    // intersection
                    coordinate id = iloc(ray.origin + coordinate(0,0,1)*t);
                    if(id.z >= resolution) break;
                    scan_lst.push_back({id.z, status});
                    voxels->voxel_list[x][y][int(id.z)]->clash_lst.insert(entity_id);
                    // set new ray
                    ray.origin = ray.origin + (t + ACCURACY_THRESHOD)*coordinate(0,0,1);
                }
                if(scan_lst.size() == 0) continue;
                // 按照cell的边界进行补全，边界上也算是与零件求交的边界
                if(scan_lst[0].second == OUT){
                    scan_lst.insert(scan_lst.begin(), {0, IN});
                    voxels->voxel_list[x][y][0]->clash_lst.insert(entity_id);
                }
                if(scan_lst.back().second == IN){
                    scan_lst.push_back({resolution-1, OUT});
                    voxels->voxel_list[x][y][15]->clash_lst.insert(entity_id);
                } 
                for(size_t i = 0; i < scan_lst.size(); i+=2){
                    for(size_t id = scan_lst[i].first+1; id < scan_lst[i+1].first; id++){
                        // fill
                        voxels->voxel_list[x][y][id]->inner_lst.insert(entity_id);
                    }
                }
            }
        }
    }
}

void Ohm_slice::Cell::raycast_dexel(const std::vector<Entity>& entities, AXIS axis){
    // clash_lst.size() > 0 
    // 要和所有的entity求交，并存储在dexel中
    Dexels* dexels = nullptr;
    coordinate direction;
    if(axis == X){
        dexels = dexels_x;
        direction = coordinate(1, 0, 0);
    }else if(axis == Y){
        dexels = dexels_y;
        direction = coordinate(0, 1, 0);
    }else if(axis == Z){
        dexels = dexels_z;
        direction = coordinate(0, 0, 1);
    }
    size_t resolution = dexels->resolution;
    coordinate step_lens = steps(resolution, resolution, resolution);

    T_NUMBER t = std::numeric_limits<T_NUMBER>::max();
    Status status;
    for(size_t x = 0; x < resolution; x++){
        for(size_t y = 0; y < resolution; y++){
            for(size_t entity_id : clash_lst){
                Ray ray;
                if(axis == X) ray = Ray(coordinate(0, x+0.5, y+0.5).times(step_lens) + LeftDown(), X);
                if(axis == Y) ray = Ray(coordinate(x+0.5, 0, y+0.5).times(step_lens) + LeftDown(), Y);
                if(axis == Z) ray = Ray(coordinate(x+0.5, y+0.5, 0).times(step_lens) + LeftDown(), Z);
                while (true){
                    t = ray.Intersect(entities[entity_id], status);
                    // no intersection
                    if(t == std::numeric_limits<T_NUMBER>::max()) break;
                    // intersection
                    coordinate intersec = ray.origin + direction * t;
                    if(axis == X && intersec.x >= RightUp().x) break;
                    if(axis == Y && intersec.y >= RightUp().y) break;
                    if(axis == Z && intersec.z >= RightUp().z) break;
                    if(dexels->dexels[entity_id][x][y].scan_lst.size() > 0){ // 由于每次是生成了新的光线，因此需要将t累加
                        T_NUMBER end_t = dexels->dexels[entity_id][x][y].scan_lst[dexels->dexels[entity_id][x][y].scan_lst.size() - 1].first;
                        dexels->dexels[entity_id][x][y].scan_lst.push_back({t + end_t, status});
                    }else{
                        dexels->dexels[entity_id][x][y].scan_lst.push_back({t, status});
                    } 
                    // set new ray
                    ray.origin = ray.origin + (t + ACCURACY_THRESHOD) * direction;
                }
                if(dexels->dexels[entity_id][x][y].scan_lst.size() == 0) continue;
                // 按照cell的边界进行补全，边界上也算是与零件求交的边界
                if(dexels->dexels[entity_id][x][y].scan_lst[0].second == OUT){ //在开头插入值
                    dexels->dexels[entity_id][x][y].scan_lst.insert(dexels->dexels[entity_id][x][y].scan_lst.begin(), {0, IN});
                }
                if(dexels->dexels[entity_id][x][y].scan_lst.back().second == IN){ //在结尾插入值, 同时得防止id转换出问题
                    if(axis == X) dexels->dexels[entity_id][x][y].scan_lst.push_back({RightUp().x - LeftDown().x - ACCURACY_THRESHOD , OUT});
                    if(axis == Y) dexels->dexels[entity_id][x][y].scan_lst.push_back({RightUp().y - LeftDown().y - ACCURACY_THRESHOD , OUT});
                    if(axis == Z) dexels->dexels[entity_id][x][y].scan_lst.push_back({RightUp().z - LeftDown().z - ACCURACY_THRESHOD, OUT});
                }
            }
        }
    }
}

void Ohm_slice::Cell::dexel_to_voxel(){
    int resolution = voxels->res();
    coordinate step_lens = steps(resolution, resolution, resolution);
    short entity_id;
    Dexel** dexels;
    // X
    for(const auto& pair : dexels_x->dexels){
        entity_id = pair.first;
        dexels = pair.second;
        for(size_t y = 0; y < resolution; y++){
            for(size_t z = 0; z < resolution; z++){
                coordinate origin = coordinate(0, y+0.5, z+0.5).times(step_lens) + LeftDown();
                for(size_t i = 0; i < dexels[y][z].scan_lst.size(); i+=2){
                    // 遍历dexel的intervals，进出成对，因此两个一组
                    T_INDEX start = iloc(origin + coordinate(1,0,0) * dexels[y][z].scan_lst[i].first).x;
                    T_INDEX end = iloc(origin + coordinate(1,0,0) * dexels[y][z].scan_lst[i+1].first).x;
                    voxels->voxel_list[start][y][z]->clash_lst.insert(entity_id);
                    voxels->voxel_list[start][y][z]->front =true;
                    voxels->voxel_list[end][y][z]->clash_lst.insert(entity_id);
                    voxels->voxel_list[end][y][z]->back =true;
                    for(size_t id = start + 1; id < end; id++){
                        // fill
                        voxels->voxel_list[id][y][z]->inner_lst.insert(entity_id);
                        voxels->voxel_list[id][y][z]->front = true;
                        voxels->voxel_list[id][y][z]->back = true;
                    }
                }
            }
        }
    }
    // Y
    for(const auto& pair : dexels_y->dexels){
        entity_id = pair.first;
        dexels = pair.second;
        for(size_t x = 0; x < resolution; x++){
            for(size_t z = 0; z < resolution; z++){
                coordinate origin = coordinate(x+0.5, 0, z+0.5).times(step_lens) + LeftDown();
                for(size_t i = 0; i < dexels[x][z].scan_lst.size(); i+=2){
                    T_INDEX start = iloc(origin + coordinate(0,1,0) * dexels[x][z].scan_lst[i].first).y;
                    T_INDEX end = iloc(origin + coordinate(0,1,0) * dexels[x][z].scan_lst[i+1].first).y;
                    voxels->voxel_list[x][start][z]->clash_lst.insert(entity_id);
                    voxels->voxel_list[x][start][z]->right = true;
                    voxels->voxel_list[x][end][z]->clash_lst.insert(entity_id);
                    voxels->voxel_list[x][end][z]->left = true;
                    for(size_t id = start + 1; id < end; id++){
                        voxels->voxel_list[x][id][z]->inner_lst.insert(entity_id);
                        voxels->voxel_list[x][id][z]->left = true;
                        voxels->voxel_list[x][id][z]->right = true;
                    }
                }
            }
        }
    }
    // Z
    for(const auto& pair : dexels_z->dexels){
        entity_id = pair.first;
        dexels = pair.second;
        for(size_t x = 0; x < resolution; x++){
            for(size_t y = 0; y < resolution; y++){
                coordinate origin = coordinate(x+0.5, y+0.5, 0).times(step_lens) + LeftDown();
                for(size_t i = 0; i < dexels[x][y].scan_lst.size(); i+=2){
                    T_INDEX start = iloc(origin + coordinate(0,0,1) * dexels[x][y].scan_lst[i].first).z;
                    T_INDEX end = iloc(origin + coordinate(0,0,1) * dexels[x][y].scan_lst[i+1].first).z;
                    voxels->voxel_list[x][y][start]->clash_lst.insert(entity_id); 
                    voxels->voxel_list[x][y][start]->up = true;
                    voxels->voxel_list[x][y][end]->clash_lst.insert(entity_id); 
                    voxels->voxel_list[x][y][end]->down = true;
                    for(size_t id = start + 1; id < end; id++){
                        // fill
                        voxels->voxel_list[x][y][id]->inner_lst.insert(entity_id);
                        voxels->voxel_list[x][y][id]->down = true;
                        voxels->voxel_list[x][y][id]->up = true;
                    }
                }
            }
        }
    }
   
}


