#include "Cell.h"
#include "my_macros.h"
#include <cmath>
#include <algorithm>

void Ohm_slice::Cell::refine_to_voxels(int res){
    voxels = new Voxels(res, *this);
}

void Ohm_slice::Cell::refine_to_dexels(int res){
    dexels = new Dexels(res, clash_lst);
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
                Ray ray(coordinate(x+0.5, y+0.5, z).times(step_lens) + LeftDown());
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
                    voxels->voxel_list[x][y][int(id.z)]->clash_lst.push_back(entity_id);
                    // set new ray
                    ray.origin = ray.origin + (t + ACCURACY_THRESHOD)*coordinate(0,0,1);
                }
                if(scan_lst.size() == 0) continue;
                // 按照cell的边界进行补全，边界上也算是与零件求交的边界
                if(scan_lst[0].second == OUT){
                    scan_lst.insert(scan_lst.begin(), {0, IN});
                    voxels->voxel_list[x][y][0]->clash_lst.push_back(entity_id);
                }
                if(scan_lst.back().second == IN){
                    scan_lst.push_back({resolution-1, OUT});
                    voxels->voxel_list[x][y][15]->clash_lst.push_back(entity_id);
                } 
                for(size_t i = 0; i < scan_lst.size(); i+=2){
                    for(size_t id = scan_lst[i].first+1; id < scan_lst[i+1].first; id++){
                        // fill
                        voxels->voxel_list[x][y][id]->inner_lst.push_back(entity_id);
                    }
                }
            }
        }
    }
}

void Ohm_slice::Cell::raycast_dexel(const std::vector<Entity>& entities){
    // clash_lst.size() > 0 
    // 要和所有的entity求交，并存储在dexel中
    size_t resolution = dexels->resolution;
    coordinate step_lens = steps(resolution, resolution, resolution);
    const size_t z = 0;
    T_NUMBER t = std::numeric_limits<T_NUMBER>::max();
    Status status;
    // std::vector<std::pair<size_t, Status>> scan_lst; // 扫描线的交点信息
    for(size_t x = 0; x < resolution; x++){
        for(size_t y = 0; y < resolution; y++){
            for(size_t entity_id : clash_lst){
                Ray ray(coordinate(x+0.5, y+0.5, z).times(step_lens) + LeftDown());
                // scan_lst.clear();
                while (true){
                    t = ray.Intersect(entities[entity_id], status);
                    // no intersection
                    if(t == std::numeric_limits<T_NUMBER>::max()) break;
                    // intersection
                    coordinate intersec = ray.origin + coordinate(0,0,1)*t;
                    if(intersec.z >= RightUp().z) break;
                    
                    dexels->dexels[entity_id][x][y].scan_lst.push_back({t, status});
                    // set new ray
                    ray.origin = ray.origin + (t + ACCURACY_THRESHOD)*coordinate(0,0,1);
                }
                if(dexels->dexels[entity_id][x][y].scan_lst.size() == 0) continue;
                // 按照cell的边界进行补全，边界上也算是与零件求交的边界
                if(dexels->dexels[entity_id][x][y].scan_lst[0].second == OUT){
                    dexels->dexels[entity_id][x][y].scan_lst.insert(dexels->dexels[entity_id][x][y].scan_lst.begin(), {0, IN});
                }
                if(dexels->dexels[entity_id][x][y].scan_lst.back().second == IN){
                    dexels->dexels[entity_id][x][y].scan_lst.push_back({RightUp().z - LeftDown().z, OUT});
                }
            }
        }
    }
}