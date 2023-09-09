#pragma once
#include "json_reader.h"
#include "stl_reader.h"
#include "my_macros.h"
#include <algorithm>
#include <cmath>

class coordinate;
class Box;
class triangle;
class Entity;
class BVHNode;
class Ray;
enum Status{IN, OUT};

class coordinate{
    // 作为顶点坐标和法向量的数据结构
    public:
        coordinate(){}
        coordinate(T_NUMBER x_, T_NUMBER y_, T_NUMBER z_):x(x_), y(y_), z(z_){}
        T_NUMBER x, y, z;
        coordinate(T_NUMBER* n) : x(n[0]), y(n[1]), z(n[2]){}
        coordinate(float* n) : x(n[0]), y(n[1]), z(n[2]){}
        static coordinate min(const coordinate& a, const coordinate& b) {
            return coordinate(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
        }
        static coordinate max(const coordinate& a, const coordinate& b) {
            return coordinate(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
        }
        coordinate normalize(){
            T_NUMBER len = x * x + y * y + z * z;
            len = sqrt(len);
            return coordinate(x/len, y/len, z/len);
        }

        bool is_nan(){
            return std::isnan(x) && std::isnan(y) && std::isnan(z);
        } 

        // 重载加法运算符
        coordinate operator+(const coordinate& other) const {
            return coordinate(x + other.x, y + other.y, z + other.z);
        }
        // 重载减法运算符
        coordinate operator-(const coordinate& other) const {
            return coordinate(x - other.x, y - other.y, z - other.z);
        }
        // 重载乘法运算符（数乘）
        coordinate operator*(T_NUMBER scalar) const {
            return coordinate(x * scalar, y * scalar, z * scalar);
        }
        // 重载乘法运算符（点乘）
        T_NUMBER operator*(const coordinate& other) const {
            return x * other.x + y * other.y + z * other.z;
        }
        // 重载乘法运算符（叉乘）
        coordinate operator^(const coordinate& other) const {
            return coordinate(y * other.z - z * other.y,
                                z * other.x - x * other.z,
                                x * other.y - y * other.x);
        }
        // 广播机制
        coordinate times(const coordinate & other) {
            return coordinate(x * other.x, y * other.y, z * other.z);
        }
        friend std::ostream& operator<<(std::ostream& os, const coordinate& point) {
            os << "coordinate: (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
            return os;
        }
};

// 重载数乘运算符（满足数乘的前后顺序）
template <typename T>
coordinate operator*(T scalar, const coordinate& vec) {
    return vec * scalar;
}

class Box{
    public:
        Box(){}
        Box(coordinate ld, coordinate ru):leftdown(ld), rightup(ru){}
        Box operator+(const Box& other) { // 合并操作
            coordinate new_ld = coordinate::min(leftdown, other.leftdown);
            coordinate new_ru = coordinate::max(rightup, other.rightup);
            return Box(new_ld, new_ru);
        }
        coordinate steps(const size_t res_x, const size_t res_y, const size_t res_z){
            // 输出细分之后子Box的长宽高
            T_NUMBER x_step = (rightup.x - leftdown.x)/res_x;
            T_NUMBER y_step = (rightup.y - leftdown.y)/res_y;
            T_NUMBER z_step = (rightup.z - leftdown.z)/res_z;
            return coordinate(x_step, y_step, z_step);
        }
        const coordinate LeftDown(){return leftdown;}
        const coordinate RightUp(){return rightup;}
        bool clash(Box box);
        bool clash(triangle tri);
        bool clash(Entity entity);
        void projectOnAxis(const coordinate& axis, T_NUMBER& min, T_NUMBER& max);
        bool overlap(T_NUMBER min1, T_NUMBER max1, T_NUMBER min2, T_NUMBER max2);
        friend std::ostream& operator<<(std::ostream& os, const Box& box) { // 重载输出
            os << "Leftdown: (" << box.leftdown.x << ", " << box.leftdown.y << ", " << box.leftdown.z << ")  ";
            os << "RightUp: (" << box.rightup.x << ", " << box.rightup.y << ", " << box.rightup.z << ")" << std::endl;
            return os;
        }
    private:
        coordinate rightup, leftdown;
};

class triangle{
    // 面片信息包含三角形的顶点和法向
    public:
        triangle(coordinate p1, coordinate p2, coordinate p3, coordinate n)
        : point1(p1), point2(p2), point3(p3), normal(n){};
        triangle(coordinate p1, coordinate p2, coordinate p3): point1(p1), point2(p2), point3(p3){
            coordinate n = (p2-p1)^(p3-p2);
            T_NUMBER len = sqrt(n*n);
            normal = n*(1/len);
        };
        coordinate point1, point2, point3;
        coordinate normal;
        void projectOnAxis(const coordinate& axis, T_NUMBER& min, T_NUMBER& max);
        // 重载输出运算符
        friend std::ostream& operator<<(std::ostream& os, const triangle& tri) {
            os << "Point1: (" << tri.point1.x << ", " << tri.point1.y << ", " << tri.point1.z << ")" << std::endl;
            os << "Point2: (" << tri.point2.x << ", " << tri.point2.y << ", " << tri.point2.z << ")" << std::endl;
            os << "Point3: (" << tri.point3.x << ", " << tri.point3.y << ", " << tri.point3.z << ")" << std::endl;
            os << "Normal: (" << tri.normal.x << ", " << tri.normal.y << ", " << tri.normal.z << ")" << std::endl;
            return os;
        }
};

class Entity{
    // 存储面片信息 + 是否是金属
    public:
        Entity(bool is_m, std::vector<triangle> tri):is_metal(is_m), triangles(tri){};
        int count_triangle(){return triangles.size();};
        void build_BVH();
        bool clash(Box box);
        // bool clash(Ray ray);
        bool is_metal = false;
    // private:
        std::vector<triangle> triangles;
        BVHNode* BVHTree; 
};

class BVHNode {
    public:
        BVHNode() : left(nullptr), right(nullptr), tri(nullptr) {}
        // 构建 BVH 树的递归函数
        static BVHNode* build(std::vector<triangle>& triangles, int start, int end);
        static Box calculateBoundingBox(const triangle& tri);
        static BVHNode* buildLeafNode(triangle* tri);
        bool clash(Box box);
        T_NUMBER Intersect(Ray ray, Status& status);
        static void print(BVHNode* root, int level = 0);
    private:
        Box bounding_box;
        BVHNode* left;
        BVHNode* right;
        triangle* tri;
   
};

class Ray{
    // every ray is cast from xy plane to positive z direction
    public:
        coordinate origin;
        T_NUMBER t;
        Ray(coordinate origin_, T_NUMBER t_ = 0): origin(origin_), t(t_){}
        bool clash(Box box){
            if(origin.z > box.RightUp().z) return false;
            // 判断xy坐标是否落在投影平面内
            if(origin.x < box.LeftDown().x || origin.x > box.RightUp().x) return false;
            if(origin.y < box.LeftDown().y || origin.y > box.RightUp().y) return false;
            return true;
        }
        T_NUMBER Intersect(triangle tri, Status& status){ // 可以直接投影成2维做
            auto crossProduct = [](coordinate v1, coordinate v2) {
                return v1.x * v2.y - v2.x * v1.y;
            };
            // 首先判断是否在光线的正面，这个可以由参数方程求解证明得到
            t = ((tri.point1 - origin) * tri.normal) / (tri.normal.z);
            if(t < 0) return std::numeric_limits<T_NUMBER>::max();
            // 然后在xy平面上处理是否在三角形内，用origin到三个顶点的向量之间的夹角来判断
            // 这里只用向量的xy坐标
            coordinate v1 = tri.point2 - tri.point1;
            coordinate v2 = tri.point3 - tri.point2;
            coordinate v3 = tri.point1 - tri.point3;
            T_NUMBER AB = crossProduct(v1, origin - tri.point1);
            T_NUMBER BC = crossProduct(v2, origin - tri.point2);
            T_NUMBER CA = crossProduct(v3, origin - tri.point3);
            // 如果都在三条边的左侧或者右侧，那么就在三角形内部
            if((AB <= 0 && BC <= 0 && CA <= 0) || (AB >= 0 && BC >= 0 && CA >= 0)){
                if(tri.normal.z < 0) status = IN;
                else status = OUT;
                return t;
            }
            return std::numeric_limits<T_NUMBER>::max();
        }
        T_NUMBER Intersect(Entity entity, Status& status){
            return entity.BVHTree->Intersect(*this, status);
            // T_NUMBER t = std::numeric_limits<T_NUMBER>::max();
            // for(size_t i = 0; i < entity.triangles.size(); i++){
            //     t = std::min(t, Intersect(entity.triangles[i], status));
            // }
            // return t;
        }
};

void load_entities(std::string json_file, std::string stl_file, std::vector<Entity>& entities);
