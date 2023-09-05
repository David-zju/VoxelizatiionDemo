#pragma once
#include "my_macros.h"
#include "Entities.h"
#include <vector>
#include <unordered_map>
#include "json_reader.h"

class Voxel;
class Voxels;
class Dexels;
class Dexel;

// 复用上一阶段的代码，方便做集成（还是略有不同的，到时候记得做修改，但是Cell部分应该不影响）
namespace Ohm_slice {
	struct Pixel {
		// order 代表的是材料
		// geom 代表零件id
		// region 为几何填充中patch上的区域标号
		short order, region, geom;
		Pixel() {}
		Pixel(short order1, short region1, short geom1) :order(order1), region(region1), geom(geom1){}
	};

	struct Texture {
		int width, height;
		Texture(size_t width, size_t height, int init, Pixel *buf) : buf(buf) {
			// 存储16*16个Pixel的数据
			this->width = width;
			this->height = height;
			data = new Pixel *[height];
			for (int i = 0; i < height; i++) {
				data[i] = buf + i * width;
				for (int j = 0; j < width; j++) {
					data[i][j].order = init;
					data[i][j].region = -1;
					data[i][j].geom = -1;
				}
			}
		}

		~Texture() {
			delete data;
		}

		Pixel *&operator[] (int i) {
			return data[i];
		}
		Pixel* GetPixel(size_t h, size_t w) {
			return &data[h][w];
		}
	private:
		Pixel **data;
		Pixel *buf;
		Texture() { }
	};

	class Patch {
	public:
		Ohm_slice::Pixel *pixel_list[16][16];
		//Ohm_slice::Pixel *pixel_list;

		short metal_region_num, non_metal_region_num, patch_type;//一个patch的内环个数
		Patch() {
			metal_region_num = 0;
			non_metal_region_num = 0;
			patch_type = 0;
		}
		~Patch() {

		}
		int write(std::string loc) {
			std::ofstream outfile;
			outfile.open(loc);
			for (int j = 0; j < 16; j++) {
				for (int i = 0; i < 16; i++) {
					outfile << pixel_list[i][j]->order << " ";
				}
				outfile << "\n";
			}
			outfile.close();
			return 0;
		}
	};

	class Cell: public Box {
	public:
		int xPos, yPos, zPos; //index
		// Patch *patch_list[6]; //6个面
		// double leftDown[3];
		// double rightUp[3];
		// bool isConduction[12]; // edge
		// int unflodMetalNum;
        std::vector<short> clash_lst; // Cell的表面与哪些零件有交
		Voxels* voxels;
		Dexels* dexels;
		Cell() {
			// memset((void*)isConduction, 0, 12 * sizeof(bool));
			// unflodMetalNum = 0;
		}
		Cell(const coordinate& leftDownCoord, const coordinate& rightUpCoord, int x, int y, int z) : 
        	Box(leftDownCoord, rightUpCoord), xPos(x), yPos(y), zPos(z) {}
		~Cell() {
			// for (int i = 0; i < 6; i++) {
			// 	delete patch_list[i];
			// }
		}
		void refine_to_voxels(int res);
		void refine_to_dexels(int res);
		void raycast_voxel(const std::vector<Entity>& entities);
		void raycast_dexel(const std::vector<Entity>& entities);
		coordinate iloc(coordinate realloc);
		static std::vector<std::vector<std::vector<Cell>>> build_cell_list(std::string json_file) {
			std::vector<T_NUMBER> px, py, pz;
			get_mesh(json_file, px, py, pz); // read the data from jsonfile
			// px = {-1, 0, 1};
			// py = {-1, 0, 1};
			// pz = {-1, 0, 1};
			// 注意cell的数量比网格线少1
			// 创建一个三维的 Cell vector 存储cells的几何信息
			std::vector<std::vector<std::vector<Cell>>> cell_list;
			for (size_t i = 0; i < px.size() - 1; i++) {
				std::vector<std::vector<Cell>> y_layer;
				for (size_t j = 0; j < py.size() - 1; j++) {
					std::vector<Cell> z_layer;
					for (size_t k = 0; k < pz.size() - 1; k++) {
						coordinate ld(px[i], py[j], pz[k]);
						coordinate ru(px[i + 1], py[j + 1], pz[k + 1]);
						z_layer.push_back(Cell(ld, ru, i, j, k));
					}
					y_layer.push_back(z_layer);
				}
				cell_list.push_back(y_layer);
			}
			std::cout<<"cell_list attributes are set."<<std::endl;
			return cell_list;
		}
    };
};


class Voxel: public Box{
    //  Nodes sequence in one element
    //     5 ____________ 8
    //     /            /|       z
    //    /___________ / |       |
    //  6|            |7 |       |____ y
    //   |            |  |      /
    //   |  1         |  |4    /
    //   |            | /    x
    //   |____________|/
    //  2           3
    public:
        // Voxel(std::vector<short> lst, coordinate ld, coordinate ru, Voxel* parent_ = nullptr)
        // :Box(ld, ru), clash_lst(lst), parent(parent_){
        //     for (int i = 0; i < 2; ++i) {
		// 		for (int j = 0; j < 2; ++j) {
		// 			for (int k = 0; k < 2; ++k) {
		// 				children[i][j][k] = nullptr;
		// 			}
		// 		}
		// 	}
        // };
		Voxel(coordinate ld, coordinate ru)
        :Box(ld, ru){}

        std::vector<size_t> clash_lst; // 与哪些零件表面相交
        std::vector<size_t> inner_lst; // 在哪些零件内部
        // Voxel* parent;
        // Voxel* children[2][2][2];
		// Voxel(int res, Ohm_slice::Cell cell):Box(cell.LeftDown(), cell.RightUp()), resolution(res){ // 建立坐标映射和clash_lst
		// 	clash_lst = cell.clash_lst;
		// 	refine(resolution);
        // }
		// void refine(int refine){ // 基于分辨率进行八叉树体素化
		// 	if(refine < 2) return;
		// 	coordinate steps = this->steps(2,2,2);
		// 	for (size_t x = 0; x < 2; ++x) {
        //         for (size_t y = 0; y < 2; ++y) {
        //             for (size_t z = 0; z < 2; ++z) {
		// 				coordinate id_xyz(x,y,z);
		// 				coordinate leftdown(LeftDown() + (id_xyz.times(steps)));
		// 				coordinate rightup(LeftDown() + ((id_xyz + coordinate(1,1,1)).times(steps)));
        //                 children[x][y][z] = new Voxel(clash_lst, leftdown, rightup, this);
		// 				children[x][y][z]->refine(refine/2);
        //             }
        //         }
        //     }
		// }
		// Voxel* iloc(size_t x, size_t y, size_t z){ // 基于递归的对Voxel定位的封装
			
		// }

		// private:
		// 	size_t resolution; // 2, 4, 8, 16, 32, 64
};

class Voxels{
    public:
        Voxel**** voxel_list; // 三维指针列表
        Voxels(int res, Ohm_slice::Cell cell):resolution(res){ // 建立坐标映射
			voxel_list = new Voxel***[resolution]; // 注意内存需要动态分配
            coordinate steps = cell.steps(resolution,resolution,resolution);
            for (size_t x = 0; x < resolution; ++x) {
				voxel_list[x] = new Voxel**[resolution];
                for (size_t y = 0; y < resolution; ++y) {
					voxel_list[x][y] = new Voxel*[resolution];
                    for (size_t z = 0; z < resolution; ++z) {
						coordinate id_xyz(x,y,z);
						coordinate leftdown(cell.LeftDown() + (id_xyz.times(steps)));
						coordinate rightup(cell.LeftDown() + ((id_xyz + coordinate(1,1,1)).times(steps)));
                        voxel_list[x][y][z] = new Voxel(leftdown, rightup);
                    }
                }
            }
        }
		size_t res(){return resolution;}
    private:
        size_t resolution; // 2, 4, 8, 16, 32, 64
};

class Dexel{
	// 不同零件用不同的dexel
	public:
		// {t, status} t = [0,1]
		std::vector<std::pair<T_NUMBER, Status>> scan_lst;
		bool isInside(T_NUMBER t){
			if(scan_lst.size() == 0) return false;
			// 判断cell内部的某个点是否在某个零件内
			for(std::pair p : scan_lst){
				if(p.first > t) return bool(p.second);
			}
			throw std::runtime_error("超出Dexel范围");
		}
};

class Dexels{
	public:
		size_t resolution;
		std::unordered_map<short, Dexel**> dexels;
		Dexels(size_t res, const std::vector<short>& clash_lst) : resolution(res) {
			// 初始化 resolution 成员变量
			resolution = res;

			// 初始化 dexels 成员变量
			for (size_t clash : clash_lst) {
				dexels[clash] = new Dexel*[resolution];
				for (size_t i = 0; i < resolution; ++i) {
					dexels[clash][i] = new Dexel[resolution];
				}
			}
    	}
		~Dexels() {
        for (auto& entry : dexels) {
            for (size_t i = 0; i < resolution; ++i) {
                delete[] entry.second[i];
            }
            delete[] entry.second;
        }
    }
};
