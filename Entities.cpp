#include "Entities.h"
#include <exception>
#include "my_macros.h"
#include <algorithm>
#include <cmath>

void load_entities(std::string json_file, std::string stl_file, std::vector<Entity>& entities){
    bool print = false;
    std::vector<bool> is_metal =  get_bool_property(json_file);
    stl_reader::StlMesh <float, unsigned int> mesh (stl_file);
    entities.clear();
    for(size_t isolid = 0; isolid < mesh.num_solids(); ++isolid) {
        std::vector<triangle> triangles; // 单个零件的triangles
        triangles.clear();
        for(size_t itri = mesh.solid_tris_begin(isolid); itri < mesh.solid_tris_end(isolid); ++itri){
            // coordinate normal((float*) mesh.tri_normal (itri));
            coordinate p1((float*) mesh.tri_corner_coords (itri, 0));
            coordinate p2((float*) mesh.tri_corner_coords (itri, 1));
            coordinate p3((float*) mesh.tri_corner_coords (itri, 2));
            triangle tri(p1, p2, p3);

            triangles.push_back(tri);
            if(print){
                std::cout   << "normal of triangle " << itri << ": "
                            << "(" << tri.normal.x << ", " << tri.normal.y << ", " << tri.normal.z << ")\n";
                std::cout << "coordinates of triangle " << itri << ": ";
                std::cout << "(" << p1.x << ", " << p1.y << ", " << p1.z << ") " ;
                std::cout << "(" << p2.x << ", " << p2.y << ", " << p2.z << ") " ;
                std::cout << "(" << p3.x << ", " << p3.y << ", " << p3.z << ") " << std::endl;
            }
            
        }
        Entity entity(is_metal[isolid], triangles);
        entities.push_back(entity);
    }
    if(entities.size() != is_metal.size()){
        // throw std::runtime_error("json信息与STL信息不一致！");
    }else{
        std::cout<< "entities attributes are set." << std::endl;
    }
}

void triangle::projectOnAxis(const coordinate& axis, T_NUMBER& min, T_NUMBER& max){
    min = std::min({point1 * axis, point2 * axis, point3 * axis});
    max = std::max({point1 * axis, point2 * axis, point3 * axis});
}

void Entity::build_BVH(){
    BVHTree = BVHNode::build(triangles, 0, triangles.size() - 1);
    // BVHNode::print(BVHTree);
}

BVHNode* BVHNode::build(std::vector<triangle>& triangles, int start, int end) {
    BVHNode* node = new BVHNode();
    Box box(coordinate(0.85,-1,-1), coordinate(0.9,1,1));
    // 如果只有一个三角形，当前节点就是叶节点
    if (start == end) {
        node->bounding_box = calculateBoundingBox(triangles[start]);
        node->tri = &triangles[start];
        node->left = nullptr;
        node->right = nullptr;
        
    } else if (start + 1 == end) {
        // 如果只有两个三角形，创建左右子节点
        node->left = buildLeafNode(&triangles[start]);
        node->right = buildLeafNode(&triangles[end]);
        node->bounding_box = node->left->bounding_box + node->right->bounding_box;
        
    } else {
        // 如果有多个三角形，递归地构建 BVH 树
        int mid = (start + end) / 2;
        BVHNode* left_child = build(triangles, start, mid);
        BVHNode* right_child = build(triangles, mid + 1, end);
        node->left = left_child;
        node->right = right_child;
        node->bounding_box = left_child->bounding_box + right_child->bounding_box;
    }

    return node;
}

Box BVHNode::calculateBoundingBox(const triangle& tri) {
    // 获取三角形的顶点坐标
    const coordinate& p1 = tri.point1;
    const coordinate& p2 = tri.point2;
    const coordinate& p3 = tri.point3;
    
    // 计算最小和最大坐标值
    coordinate rightup = coordinate::max(coordinate::max(p1,p2),p3);
    coordinate leftdown = coordinate::min(coordinate::min(p1,p2),p3);
    
    // 构建包围盒对象并返回
    return Box(leftdown, rightup);
}

BVHNode* BVHNode::buildLeafNode(triangle* tri) { // 这里似乎用不了const triangle& 类型
    Box box(coordinate(0.85,-1,-1), coordinate(0.9,1,1));
    BVHNode* node = new BVHNode();
    node->bounding_box = calculateBoundingBox(*tri);
    node->tri = tri;
    node->left = nullptr;
    node->right = nullptr;
    return node;
}

bool BVHNode::clash(Box box){
    if(!bounding_box.clash(box)) return false;
    if(left != nullptr || right != nullptr){
        bool l_result = false, r_result = false;
        if(left != nullptr)l_result = left->clash(box);
        if (right != nullptr) r_result = right->clash(box);
        return (l_result || r_result);
    }
    // leaf node case
    // if(!left && !right){
        return box.clash(*tri);
        // return true;
    // }   
}

T_NUMBER BVHNode::Intersect(Ray ray, Status& status){
    // 返回的是交到的t值
    if(!ray.clash(bounding_box)){
        return std::numeric_limits<T_NUMBER>::max();
    }
    if(left != nullptr || right != nullptr){
        T_NUMBER l_result = std::numeric_limits<T_NUMBER>::max();
        T_NUMBER r_result = std::numeric_limits<T_NUMBER>::max();
        Status l_status, r_status;
        if(left != nullptr)l_result = left->Intersect(ray, l_status);
        if (right != nullptr) r_result = right->Intersect(ray, r_status);
        if(r_result < l_result) status = r_status;
        else status = l_status; 
        return (std::min(l_result, r_result));
    }
    // leaf node case
    return ray.Intersect(*tri, status);
}

// void BVHNode::print(BVHNode* root, int level){ // debug function
//     if(!root) return;
//     print(root->left, level + 1);
//     for (int i = 0; i < level; ++i) {
//         std::cout << "    ";
//     }
//     if(!root->tri) std::cout << -1 << std::endl;
//     else std::cout << root->tri->debug_id << std::endl;
    
//     print(root->right, level + 1);
// }

bool Box::clash(Box box){
    // 查看在三个方向上是否重叠
    if(rightup.x < box.LeftDown().x || leftdown.x > box.RightUp().x) return false;
    if(rightup.y < box.LeftDown().y || leftdown.y > box.RightUp().y) return false;
    if(rightup.z < box.LeftDown().z || leftdown.z > box.RightUp().z) return false;
    return true;
}

bool Box::clash(triangle tri){
    // 包围盒初判
    Box BoundingBox = BVHNode::calculateBoundingBox(tri);
    if(!clash(BoundingBox)) return false;
    //分离轴算法，xyz3个轴其实就是包围盒初判
    T_NUMBER boxMin, boxMax, triMin, triMax;
    std::vector<coordinate> axis_lst;
    axis_lst.push_back(tri.normal); // 三角形的法向
    coordinate temp[] = {coordinate(1,0,0), coordinate(0,1,0), coordinate(0,0,1)};
    for(size_t i = 0; i < 3; i++){ // 三角形的三个边与长方体三个轴构成平面的法向 共9个
        axis_lst.push_back((temp[i]^(tri.point1 - tri.point2)).normalize());
        axis_lst.push_back((temp[i]^(tri.point2 - tri.point3)).normalize());
        axis_lst.push_back((temp[i]^(tri.point3 - tri.point1)).normalize());
    }
    for(coordinate axis : axis_lst){
        if(axis.is_nan()) continue;
        Box::projectOnAxis(axis, boxMin, boxMax);
        tri.projectOnAxis(axis, triMin, triMax);
        if(!Box::overlap(boxMin, boxMax, triMin, triMax)){
            return false;
        }
    }
    return true;
}

void Box::projectOnAxis(const coordinate& axis, T_NUMBER& min, T_NUMBER& max){
    coordinate boxVertices[8] = {
        coordinate(leftdown.x, leftdown.y, leftdown.z),
        coordinate(rightup.x, leftdown.y, leftdown.z),
        coordinate(leftdown.x, rightup.y, leftdown.z),
        coordinate(rightup.x, rightup.y, leftdown.z),
        coordinate(leftdown.x, leftdown.y, rightup.z),
        coordinate(rightup.x, leftdown.y, rightup.z),
        coordinate(leftdown.x, rightup.y, rightup.z),
        coordinate(rightup.x, rightup.y, rightup.z)
    };

    min = std::numeric_limits<T_NUMBER>::max();
    max = std::numeric_limits<T_NUMBER>::lowest();

    for(size_t i = 0; i < 8; i++){
        T_NUMBER projection = boxVertices[i] * axis;
        min = std::min(min, projection);
        max = std::max(max, projection);
    }
}

bool Box::overlap(T_NUMBER min1, T_NUMBER max1, T_NUMBER min2, T_NUMBER max2) {
    return max1 >= min2 && max2 >= min1;
}

bool Box::clash(Entity entity){
    return entity.clash(*this);
}

bool Entity::clash(Box box){
    return BVHTree->clash(box);
    // for(triangle tri : triangles){
    //     if(box.clash(tri)) return true;
    // }
    // return false;
}

